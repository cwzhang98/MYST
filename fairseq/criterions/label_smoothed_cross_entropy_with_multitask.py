import math
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from omegaconf import II
from typing import Optional
from fairseq.data.data_utils import lengths_to_mask

from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion,
    label_smoothed_nll_loss
)


@dataclass
class LabelSmoothedCrossEntropyWithMultitaskConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    use_jsd: bool = field(
        default=False,
        metadata={"help": "use jsd loss or not"}
    )
    asr_task: bool = field(
        default=False,
        metadata={"help": "use asr multitask"}
    )
    use_ctc: bool = II("model.use_ctc")
    model_type: Optional[str] = II("model.model_type")


@register_criterion(
    "label_smoothed_cross_entropy_with_multitask",
    dataclass=LabelSmoothedCrossEntropyWithMultitaskConfig
)
class LabelSmoothedCrossEntropyWithMultitask(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            use_jsd=False,
            asr_task=False,
            use_ctc=False,
            model_type=""
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.use_jsd = use_jsd
        self.asr_task = asr_task
        self.use_ctc = use_ctc
        self.model_type = model_type
        self.blank_idx = task.tgt_dict.bos()
        self.eos_idx = task.tgt_dict.eos()

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], is_audio_input=True)
        st_loss, nll_loss_st = torch.tensor(0.), torch.tensor(0.)
        mt_loss, nll_loss_mt = torch.tensor(0.), torch.tensor(0.)
        asr_loss, nll_loss_asr = torch.tensor(0.), torch.tensor(0.)
        jsd_loss, qua_loss = torch.tensor(0.), torch.tensor(0.)
        if model.training:
            net_output, encoder_out = net_output
        if sample["target"] is not None:
            st_loss, nll_loss_st, lprobs_st, target = self.compute_loss(
                model, net_output, sample, reduce=reduce)
            if model.training:
                mt_loss, nll_loss_mt, lprobs_mt = self.compute_loss_mt(
                    model, sample, reduce=reduce)
                if self.asr_task:
                    if self.use_ctc:
                        ctc_padding_mask = encoder_out["ctc_padding_mask"][0] \
                            if encoder_out["ctc_padding_mask"] is not None else encoder_out["encoder_padding_mask"][0]
                        asr_loss = self.compute_loss_ctc(
                            sample,
                            encoder_out["ctc_logits"][0],
                            ctc_padding_mask,
                            reduce=reduce
                        )
                    else:
                        asr_loss, nll_loss_asr = self.compute_loss_asr(model, sample, reduce=reduce)
                if self.use_jsd:
                    jsd_loss = self.compute_loss_jsd(lprobs_st, lprobs_mt, target, reduce=reduce)
                if self.model_type != "w2v_transformer":
                    assert sample["source_lengths"] is not None
                    qua_loss = self.compute_qua_loss(encoder_out, sample["source_lengths"])

        loss = st_loss + mt_loss + asr_loss + jsd_loss + qua_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss_st.data),
            "nll_loss_mt": utils.item(nll_loss_mt.data),
            "nll_loss_asr": utils.item(nll_loss_asr.data),
            "ctc_loss": utils.item(asr_loss.data) if self.use_ctc else 0,
            "jsd_loss": utils.item(jsd_loss.data),
            "qua_loss": utils.item(qua_loss),
            "ntokens": sample["target_ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "source_ntokens": sample["source_ntokens"],
            "target_tokens": sample["target_ntokens"]
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target

    def compute_loss_mt(self, model, sample, reduce=True):
        decoder_out, _ = model(
            sample["source"],
            sample["source_lengths"],
            sample["net_input"]["prev_output_tokens"],
            is_audio_input=False
        )
        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        target = model.get_targets(sample, decoder_out)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce
        )
        return loss, nll_loss, lprobs

    def compute_loss_asr(self, model, sample, reduce=True):
        decoder_out, encoder_out = model(
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
            sample["prev_output_src_tokens"]
        )
        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        target = sample["source"]
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce
        )
        return loss, nll_loss

    def compute_loss_ctc(self, sample, ctc_logits, padding_mask, reduce=True):
        ctc_lprobs = F.log_softmax(ctc_logits, dim=-1)
        ctc_lens = ctc_lprobs.new_full((ctc_lprobs.shape[1],), ctc_lprobs.shape[0]).long()
        ctc_lens -= padding_mask.long().sum(dim=-1)
        # remove language token and eos token
        ctc_tgt, ctc_tgt_lens = sample["source"][:, self.ignore_prefix_size:], sample["source_lengths"] - 1
        ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
        ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
        eos_mask = ctc_tgt_flat.ne(self.eos_idx)
        ctc_tgt_flat = ctc_tgt_flat.masked_select(eos_mask)
        ctc_tgt_lens -= 1
        reduction = "sum" if reduce else "None"
        ctc_loss = F.ctc_loss(
            ctc_lprobs,
            ctc_tgt_flat,
            ctc_lens,
            ctc_tgt_lens,
            blank=self.blank_idx,
            reduction=reduction,
            zero_infinity=True
        )
        return ctc_loss

    def compute_loss_jsd(self, lprobs_st, lprobs_mt, target, reduce=True):
        lprobs_mix = 0.5 * (lprobs_st + lprobs_mt)
        kl_st = F.kl_div(lprobs_mix, lprobs_st, log_target=True, reduction="none").sum(-1)
        kl_mt = F.kl_div(lprobs_mix, lprobs_mt, log_target=True, reduction="none").sum(-1)
        pad_mask = target.eq(self.padding_idx)
        kl_mt.masked_fill_(pad_mask, 0.0)
        kl_st.masked_fill_(pad_mask, 0.0)
        if reduce:
            jsd_loss = 0.5 * (kl_mt.sum() + kl_st.sum())
        else:
            jsd_loss = 0.5 * (kl_mt + kl_st)
        return jsd_loss

    def compute_qua_loss(self, encoder_out, source_lengths):
        alpha_sum = torch.sum(encoder_out["alpha"], dim=-1)
        source_lengths = source_lengths.type_as(alpha_sum)
        return F.l1_loss(alpha_sum, source_lengths, reduction='sum')

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_st_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        nll_loss_mt_sum = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)
        nll_loss_asr_sum = sum(log.get("nll_loss_asr", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        qua_loss_sum = sum(log.get("qua_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_st",
                           nll_loss_st_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_mt",
                           nll_loss_mt_sum / sample_size / math.log(2), sample_size, round=3)
        if nll_loss_asr_sum > 0:
            metrics.log_scalar("nll_loss_asr",
                               nll_loss_asr_sum / sample_size / math.log(2), sample_size, round=3)
        if ctc_loss_sum > 0:
            metrics.log_scalar("ctc_loss",
                               ctc_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if jsd_loss_sum > 0:
            metrics.log_scalar("jsd_loss",
                               jsd_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if qua_loss_sum > 0:
            metrics.log_scalar("qua_loss",
                               qua_loss_sum / nsentences, sample_size, round=3)
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
