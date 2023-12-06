import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy_with_ctc import LabelSmoothedCrossEntropyWithCtcCriterionConfig

from fairseq.data.data_utils import lengths_to_mask
from fairseq.logging.meters import safe_round
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyWithCtcWithQuaCriterionConfig(
    LabelSmoothedCrossEntropyWithCtcCriterionConfig
):
    beta: float = II("model.beta")

@register_criterion(
    "label_smoothed_cross_entropy_with_ctc_with_qua",
    dataclass=LabelSmoothedCrossEntropyWithCtcWithQuaCriterionConfig
)
class LabelSmoothedCrossEntropyWithCtcWithQuaCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        ctc_weight,
        beta
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy
        )
        self.ctc_weight = ctc_weight
        self.beta = beta
        assert beta is not None, "no beta value"
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
    
    def forward(self, model, sample, reduce=True):
        net_output = model(sample["target_lengths"], **sample["net_input"])
        # smoothed loss, nll_loss
        # target pad mask apply in label_smoothed_nll_loss()
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        ctc_loss = torch.tensor(0.0).type_as(loss)

        if self.ctc_weight > 0.0:
            ctc_lprobs, ctc_lens = model.get_ctc_output(net_output) # ctc_lprobs: T x B x C
            ctc_tgt, ctc_tgt_lens = model.get_ctc_target(sample)
            # apply target pad mask
            ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
            ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
            reduction = "sum" if reduce else "None"
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = F.ctc_loss(
                    ctc_lprobs,
                    ctc_tgt_flat,
                    ctc_lens,
                    ctc_tgt_lens,
                    blank=self.blank_idx,
                    reduction=reduction,
                    zero_infinity=True
                ) * self.ctc_weight
        # Quantity Loss
        alpha_sum = torch.sum(net_output["alpha"], dim=-1) / self.beta
        target_lengths = sample["target_lengths"].type_as(alpha_sum)
        qua_loss = F.l1_loss(alpha_sum, target_lengths, reduction='sum')

        loss = (1 - self.ctc_weight) * loss + ctc_loss + qua_loss
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "qua_loss": utils.item(qua_loss.data)
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        # compute wer during validation
        if not model.training:
            import editdistance
            with torch.no_grad():
                # both B x T x C
                lprobs = model.get_normalized_probs(net_output, log_probs=True).float().contiguous().cpu()
                ctc_lprobs_t = ctc_lprobs.transpose(0, 1).float().contiguous().cpu()

                w_errs, ctc_w_errs = 0, 0
                w_len = 0
                c_err, ctc_c_err = 0, 0
                c_len = 0
                # ce wer
                for lprob, ctc_lprob, target, input_length, ctc_input_length in zip(
                        lprobs, ctc_lprobs_t, ctc_tgt, ctc_tgt_lens, ctc_lens
                    ):
                    lprob, ctc_lprob = lprob[:input_length], ctc_lprob[:ctc_input_length]
                    p = (target != self.task.target_dictionary.pad()) & \
                        (target != self.task.target_dictionary.eos())
                    targ = target[p] # remove pad and eos symbols
                    targ_units_arr = targ.tolist()
                    # unit err rate
                    # compress ctc tokens
                    toks, ctc_toks = lprob.argmax(dim=-1), ctc_lprob.argmax(dim=-1).unique_consecutive()
                    # remove blank tokens
                    pred_units_arr, pred_ctc_units_arr = toks[toks != self.blank_idx].tolist() \
                                                        , ctc_toks[ctc_toks != self.blank_idx].tolist()
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    ctc_c_err += editdistance.eval(pred_ctc_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_units = self.task.target_dictionary.string(targ)
                    targ_words = targ_units.split()
                    pred_words = self.task.target_dictionary.string(pred_units_arr).split()
                    ctc_pred_words = self.task.target_dictionary.string(pred_ctc_units_arr).split()
                    w_errs += editdistance.eval(pred_words, targ_words)
                    ctc_w_errs += editdistance.eval(ctc_pred_words, targ_words)
                    w_len += len(targ_words)
                
                logging_output["w_errors"] = w_errs
                logging_output["ctc_w_errs"] = ctc_w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["ctc_c_errs"] = ctc_c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        qua_loss = sum(log.get("qua_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "qua_loss", qua_loss / sample_size / math.log(2), sample_size, round=3
        )
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        ctc_c_errs = sum(log.get("ctc_c_errs", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_c_errs", ctc_c_errs)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        ctc_w_errs = sum(log.get("ctc_w_errs", 0) for log in logging_outputs)
        metrics.log_scalar("_ctc_w_errs", ctc_w_errs)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "ctc_uer",
                lambda meters: safe_round(
                    meters["_ctc_c_errs"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "ctc_wer",
                lambda meters: safe_round(
                    meters["_ctc_w_errs"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
