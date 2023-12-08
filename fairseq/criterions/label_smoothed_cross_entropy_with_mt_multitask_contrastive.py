import math
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from fairseq import metrics, utils
from typing import Optional
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy_with_mt_multitask import (
    LabelSmoothedCrossEntropyWithMtMultitask,
    LabelSmoothedCrossEntropyWithMtMultitaskConfig,
    label_smoothed_nll_loss
)


@dataclass
class LabelSmoothedCrossEntropyWithMtMultitaskContrastiveConfig(
    LabelSmoothedCrossEntropyWithMtMultitaskConfig
):
    contrast_granularity: str = field(
        default="fine",
        metadata={"help": "granularity of single loss, could be `fine` or `coarse`"}
    )
    contrast_level: str = field(
        default="low",
        metadata={"help": "implement contrastive loss on `low` level or `high` level representation"}
    )
    contrastive_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "the temperature in the contrastive loss"}
    )
    contrastive_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of contrastive loss"}
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_mt_multitask_contrastive",
    dataclass=LabelSmoothedCrossEntropyWithMtMultitaskContrastiveConfig
)
class LabelSmoothedCrossEntropyWithMtMultitaskContrastiveCriterion(
    LabelSmoothedCrossEntropyWithMtMultitask
):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            use_jsd=False,
            contrast_granularity=None,
            contrast_level=None,
            contrastive_temperature=1.0,
            contrastive_weight=1.0
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy, use_jsd)
        self.contrast_granularity = contrast_granularity
        self.contrast_level = contrast_level
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], is_audio_input=True)
        st_loss, nll_loss_st = torch.tensor(0.), torch.tensor(0.)
        mt_loss, nll_loss_mt = torch.tensor(0.), torch.tensor(0.)
        jsd_loss, contrastive_loss = torch.tensor(0.), torch.tensor(0.)
        if model.training:
            net_output, encoder_out_st = net_output
        if sample["target"] is not None:
            st_loss, nll_loss_st, lprobs_st, target = self.compute_loss(
                model, net_output, sample, reduce=reduce)
            if model.training:
                mt_loss, nll_loss_mt, lprobs_mt, encoder_out_mt = self.compute_loss_mt(
                    model, sample, reduce=reduce)
                if self.use_jsd:
                    jsd_loss = self.compute_loss_jsd(lprobs_st, lprobs_mt, target, reduce=reduce)
                if self.contrastive_weight > 0.0:
                    contrastive_loss = self.compute_contrastive_loss(
                        model, sample, encoder_out_st, encoder_out_mt)
        loss = st_loss + mt_loss + jsd_loss + self.contrastive_weight * contrastive_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss_st.data),
            "nll_loss_mt": utils.item(nll_loss_mt.data),
            "jsd_loss": utils.item(jsd_loss.data),
            "contrastive_loss": utils.item(contrastive_loss.data),
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

    def compute_loss_mt(self, model, sample, reduce=True):
        decoder_out, encoder_out = model(
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
        return loss, nll_loss, lprobs, encoder_out

    def compute_contrastive_loss(self, model, sample, encoder_out_st, encoder_out_mt, reduce=True):
        if self.contrast_granularity == 'coarse':
            if self.contrast_level == 'high':
                # contrast on encoder hidden states
                seq_hidden_st = self.get_seq_hidden(
                    encoder_out_st["encoder_out"], encoder_out_st["encoder_padding_mask"][0])
                seq_hidden_mt = self.get_seq_hidden(
                    encoder_out_mt["encoder_out"], encoder_out_mt["encoder_padding_mask"][0])
                ...
            elif self.contrast_level == 'low':
                # contrast on embeddings
                # forward embedding
                audio_out = model.encoder.encode_audio(
                    sample["net_input"]["src_tokens"],
                    sample["net_input"]["src_lengths"]
                )
                audio_embed, padding_mask_st = audio_out[0], audio_out[1]
                text_embed, padding_mask_mt = model.encoder.encode_text(sample["source"])
                seq_hidden_st = self.get_seq_hidden(
                    [audio_embed], padding_mask_st)
                seq_hidden_mt = self.get_seq_hidden(
                    [text_embed], padding_mask_mt)
            else:
                raise ValueError(
                    f"Excepted `contrast_level` should be `low` or `high`, got {self.contrast_level}"
                )
            sim_matrix = F.cosine_similarity(
                seq_hidden_st.unsqueeze(0),
                seq_hidden_mt.unsqueeze(1),
                dim=-1
            )
            sim_matrix /= self.contrastive_temperature
            contrastive_loss = -torch.nn.LogSoftmax(0)(sim_matrix).diag()
            del sim_matrix
            del seq_hidden_st
            del seq_hidden_mt
        elif self.contrast_granularity == 'fine':
            ...
        else:
            raise ValueError(
                f"Excepted `contrast_granularity` should be `fine` or `coarse`, got {self.contrast_granularity}"
            )
        return contrastive_loss.sum() if reduce else contrastive_loss

    def get_seq_hidden(self, encoder_states_list, padding_mask):
        padding_mask = ~padding_mask
        B, C = encoder_states_list[0].size(1), encoder_states_list[0].size(2)
        seq_hidden = encoder_states_list[0].new_zeros((B, C))
        if len(encoder_states_list) > 1:
            for states in encoder_states_list:
                states = states.transpose(0, 1)
                seq_hidden += ((states * padding_mask.unsqueeze(-1)).sum(1) /
                               padding_mask.sum(1).unsqueeze(-1))  # B x C
            seq_hidden /= len(encoder_states_list)
        else:
            assert len(encoder_states_list) == 1
            seq_hidden = ((encoder_states_list[0].transpose(0, 1) * padding_mask.unsqueeze(-1)).sum(1) /
                          padding_mask.sum(1).unsqueeze(-1))
        return seq_hidden

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        super().reduce_metrics(logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if contrastive_loss_sum > 0:
            metrics.log_scalar("contrastive_loss",
                               contrastive_loss_sum / sample_size / math.log(2), sample_size, round=3)
