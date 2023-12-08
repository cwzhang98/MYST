import math
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion,
    label_smoothed_nll_loss
)


@dataclass
class LabelSmoothedCrossEntropyWithMtMultitaskConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    use_jsd: bool = field(
        default=False,
        metadata={"help": "use jsd loss or not"}
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_mt_multitask",
    dataclass=LabelSmoothedCrossEntropyWithMtMultitaskConfig
)
class LabelSmoothedCrossEntropyWithMtMultitask(
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
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.use_jsd = use_jsd

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], is_audio_input=True)
        st_loss, nll_loss_st = torch.tensor(0.), torch.tensor(0.)
        mt_loss, nll_loss_mt = torch.tensor(0.), torch.tensor(0.)
        jsd_loss = torch.tensor(0.)
        if model.training:
            net_output, _ = net_output
        if sample["target"] is not None:
            st_loss, nll_loss_st, lprobs_st, target = self.compute_loss(
                model, net_output, sample, reduce=reduce)
            if model.training:
                mt_loss, nll_loss_mt, lprobs_mt = self.compute_loss_mt(
                    model, sample, reduce=reduce)
                if self.use_jsd:
                    jsd_loss = self.compute_loss_jsd(lprobs_st, lprobs_mt, target, reduce=reduce)

        loss = st_loss + mt_loss + jsd_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss_st.data),
            "nll_loss_mt": utils.item(nll_loss_mt.data),
            "jsd_loss": utils.item(jsd_loss.data),
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

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_st_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        nll_loss_mt_sum = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_st",
                           nll_loss_st_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_mt",
                           nll_loss_mt_sum / sample_size / math.log(2), sample_size, round=3)
        if jsd_loss_sum > 0:
            metrics.log_scalar("jsd_loss",
                               jsd_loss_sum / sample_size / math.log(2), sample_size, round=3)
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
