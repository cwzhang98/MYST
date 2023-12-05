import math
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy_with_mt_multitask import (
    LabelSmoothedCrossEntropyWithMtMultitaskConfig,
    LabelSmoothedCrossEntropyWithMtMultitask
)
@register_criterion(
    "label_smoothed_cross_entropy_with_mt_multitask_with_qua", 
    dataclass=LabelSmoothedCrossEntropyWithMtMultitaskConfig
)
class LabelSmoothedCrossEntropyWithMtMultitaskWithQua(
    LabelSmoothedCrossEntropyWithMtMultitask
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        use_jsd=False
        ):
        super().__init__(task, sentence_avg, label_smoothing,ignore_prefix_size, report_accuracy, use_jsd)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], is_audio_input=True)
        st_loss, nll_loss_st = torch.tensor(0.), torch.tensor(0.)
        mt_loss, nll_loss_mt = torch.tensor(0.), torch.tensor(0.)
        jsd_loss, qua_loss = torch.tensor(0.), torch.tensor(0.)
        if model.training:
            net_output, encoder_out = net_output
        if sample["target"] is not None:
            st_loss, nll_loss_st, lprobs_st, probs_st, target = self.compute_loss(
                model, net_output, sample, reduce=reduce)
            if model.training:
                mt_loss, nll_loss_mt, lprobs_mt, probs_mt = self.compute_loss_mt(
                    model, sample, reduce=reduce)
                # qua loss
                if sample["net_input"]["transcript_lengths"] is not None:
                    alpha_sum = torch.sum(encoder_out["alpha"], dim=-1) \
                        / model.encoder.w2v_model.beta
                    source_lengths = sample["source_lengths"].type_as(alpha_sum)
                    qua_loss = F.l1_loss(alpha_sum, source_lengths, reduction='sum')

                if self.use_jsd:
                    lprobs_mix = 0.5 * (lprobs_st + lprobs_mt)
                    jsd_loss = self.compute_loss_jsd(lprobs_mix, probs_st, probs_mt, target, reduce=reduce)
            
        loss = st_loss + mt_loss + jsd_loss + qua_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss_st.data),
            "nll_loss_mt": utils.item(nll_loss_mt.data),
            "jsd_loss": utils.item(jsd_loss.data),
            "qua_loss": utils.item(qua_loss.data),
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
        super().reduce_metrics(logging_outputs)
        qua_loss_sum = sum(log.get("qua_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if qua_loss_sum > 0:
            metrics.log_scalar("qua_loss",
                            qua_loss_sum / sample_size / math.log(2), sample_size, round=3)