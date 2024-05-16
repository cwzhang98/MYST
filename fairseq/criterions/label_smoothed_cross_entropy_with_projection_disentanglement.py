import math
import numpy as np
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.criterions.label_smoothed_cross_entropy_with_multitask import (
    LabelSmoothedCrossEntropyWithMultitask,
    label_smoothed_nll_loss
)


@dataclass
class LabelSmoothedCrossEntropyWithProjectionDisentanglementConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    consistency_weight: float = field(
        default=1.0,
        metadata={"help": "weight of consistency loss"}
    )
    jsd_weight: float = field(
        default=1.0,
        metadata={"help": "weight of jsd loss"}
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_projection_disentanglement",
    dataclass=LabelSmoothedCrossEntropyWithProjectionDisentanglementConfig
)
class LabelSmoothedCrossEntropyWithProjectionDisentanglement(
    LabelSmoothedCrossEntropyWithMultitask
):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            consistency_weight=1.0,
            jsd_weight=1.0
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.consistency_weight = consistency_weight
        self.jsd_weight = jsd_weight

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], is_audio_input=True)
        st_loss, nll_loss_st = torch.tensor(0.), torch.tensor(0.)
        st_loss_aug, nll_loss_st_aug = torch.tensor(0.), torch.tensor(0.)
        mt_loss, nll_loss_mt = torch.tensor(0.), torch.tensor(0.)
        asr_loss, jsd_loss = torch.tensor(0.), torch.tensor(0.)
        spk_loss, noise_loss, consistency_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        if model.training:
            net_output, encoder_out = net_output
            if sample["net_input_aug"]["src_tokens"] is not None:
                net_output_aug, encoder_out_aug = model(**sample["net_input_aug"], is_audio_input=True)
        if sample["target"] is not None:
            st_loss, nll_loss_st, lprobs_st = self.compute_loss(
                model, net_output, sample, reduce=reduce)
            if model.training and sample["net_input_aug"]["src_tokens"] is not None:
                st_loss_aug, nll_loss_st_aug, lprobs_st_aug = self.compute_loss(
                    model, net_output_aug, sample, reduce=reduce)
                mt_loss, nll_loss_mt, lprobs_mt = self.compute_loss_mt(
                    model, sample, reduce=reduce)
                ctc_lprobs = F.log_softmax(encoder_out["ctc_logits"], dim=-1).transpose(0, 1)
                asr_loss = self.compute_loss_ctc(
                    sample,
                    ctc_lprobs,
                    encoder_out["ctc_padding_mask"],
                    reduce=reduce
                )
                jsd_loss = (
                    self.compute_loss_jsd(lprobs_st, lprobs_mt, sample['target'], reduce=reduce) +
                    self.compute_loss_jsd(lprobs_st_aug, lprobs_mt, sample['target'], reduce=reduce)
                ) / 2
                spk_loss = (
                   F.cross_entropy(encoder_out["speaker_logits"].float(), sample["speaker_ids"],
                                   reduction='sum', label_smoothing=self.eps) +
                   F.cross_entropy(encoder_out_aug["speaker_logits"].float(), sample["speaker_ids"],
                                   reduction='sum', label_smoothing=self.eps)
                ) / 2
                if sample["noises"] is not None:
                    noise_target = torch.full_like(sample["noises"], 4)
                    noise_loss = (
                        F.cross_entropy(encoder_out["noise_logits"].float(), noise_target,
                                        reduction='sum', label_smoothing=self.eps) * 0.5 +
                        F.cross_entropy(encoder_out_aug["noise_logits"].float(), sample["noises"],
                                        reduction='sum')
                    ) / 2
                consistency_loss = F.mse_loss(encoder_out["purified_feats"].mean(dim=0).float(),
                                              encoder_out_aug["purified_feats"].mean(dim=0).float(), reduction="sum")
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]
        )
        loss = st_loss + mt_loss + asr_loss + jsd_loss + noise_loss + spk_loss + st_loss_aug + self.consistency_weight * consistency_loss

        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss_st.data),
            "nll_loss_mt": utils.item(nll_loss_mt.data),
            "asr_loss": utils.item(asr_loss.data),
            "jsd_loss": utils.item(jsd_loss.data),
            "spk_loss": utils.item(spk_loss.data),
            "nll_loss_st_aug": utils.item(nll_loss_st_aug.data),
            "noise_loss": utils.item(noise_loss.data),
            "consistency_loss": utils.item(consistency_loss.data),
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


    @classmethod
    def reduce_metrics(cls, logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_st_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        nll_loss_mt_sum = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        asr_loss_sum = sum(log.get("asr_loss", 0) for log in logging_outputs)
        spk_loss_sum = sum(log.get("spk_loss", 0) for log in logging_outputs)
        nll_loss_st_aug_sum = sum(log.get("nll_loss_st_aug", 0) for log in logging_outputs)
        noise_loss_sum = sum(log.get("noise_loss", 0) for log in logging_outputs)
        consistency_loss_sum = sum(log.get("consistency_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_st",
                           nll_loss_st_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_mt",
                           nll_loss_mt_sum / sample_size / math.log(2), sample_size, round=3)
        if asr_loss_sum > 0:
            metrics.log_scalar("asr_loss",
                               asr_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if jsd_loss_sum > 0:
            metrics.log_scalar("jsd_loss",
                               jsd_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if spk_loss_sum > 0:
            metrics.log_scalar("spk_loss",
                               spk_loss_sum / nsentences / math.log(2), sample_size, round=3)
        if nll_loss_st_aug_sum > 0:
            metrics.log_scalar("nll_loss_st_aug",
                               nll_loss_st_aug_sum / sample_size / math.log(2), sample_size, round=3)
        if noise_loss_sum > 0:
            metrics.log_scalar("noise_loss",
                               noise_loss_sum / nsentences / math.log(2), round=3)
        if consistency_loss_sum > 0:
            metrics.log_scalar("consistency_loss",
                               consistency_loss_sum / nsentences / math.log(2), sample_size, round=3)
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
