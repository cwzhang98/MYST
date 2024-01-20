import math
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from fairseq import metrics, utils
from typing import Optional
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy_with_multitask import (
    LabelSmoothedCrossEntropyWithMultitask,
    LabelSmoothedCrossEntropyWithMultitaskConfig,
    label_smoothed_nll_loss
)


@dataclass
class LabelSmoothedCrossEntropyWithMultitaskContrastiveConfig(
    LabelSmoothedCrossEntropyWithMultitaskConfig
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
    use_dual_ctr: Optional[bool] = field(
        default=False,
        metadata={"help": "if we want to use dual contrastive loss"}
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_multitask_contrastive",
    dataclass=LabelSmoothedCrossEntropyWithMultitaskContrastiveConfig
)
class LabelSmoothedCrossEntropyWithMultitaskContrastiveCriterion(
    LabelSmoothedCrossEntropyWithMultitask
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
            contrastive_weight=1.0,
            use_dual_ctr=False,
            asr_task=False,
            use_ctc=False
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size,
                         report_accuracy, use_jsd, asr_task, use_ctc)
        self.contrast_granularity = contrast_granularity
        self.contrast_level = contrast_level
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight
        self.use_dual_ctr = use_dual_ctr

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], is_audio_input=True)
        st_loss, nll_loss_st = torch.tensor(0.), torch.tensor(0.)
        mt_loss, nll_loss_mt = torch.tensor(0.), torch.tensor(0.)
        asr_loss, nll_loss_asr = torch.tensor(0.), torch.tensor(0.)
        jsd_loss, contrastive_loss = torch.tensor(0.), torch.tensor(0.)
        if model.training:
            net_output, encoder_out_st = net_output
        if sample["target"] is not None:
            st_loss, nll_loss_st, lprobs_st, target = self.compute_loss(
                model, net_output, sample, reduce=reduce)
            if model.training:
                mt_loss, nll_loss_mt, lprobs_mt, encoder_out_mt = self.compute_loss_mt(
                    model, sample, reduce=reduce)
                if self.asr_task:
                    if self.use_ctc:
                        asr_loss = self.compute_loss_ctc(model, sample, reduce=reduce)
                    else:
                        asr_loss, nll_loss_asr = self.compute_loss_asr(model, sample, reduce=reduce)
                if self.use_jsd:
                    jsd_loss = self.compute_loss_jsd(lprobs_st, lprobs_mt, target, reduce=reduce)
                if self.contrastive_weight > 0.0:
                    contrastive_loss = self.compute_contrastive_loss(
                        model, sample, encoder_out_st, encoder_out_mt,
                        self.contrast_granularity, self.contrast_level, reduce=reduce)

        loss = st_loss + mt_loss + asr_loss + jsd_loss + self.contrastive_weight * contrastive_loss
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

    def compute_contrastive_loss(
            self,
            model,
            sample,
            encoder_out_st,
            encoder_out_mt,
            contrast_granularity,
            contrast_level,
            reduce=True,
    ):
        assert contrast_granularity == 'coarse' or contrast_granularity == 'fine'
        assert contrast_level == 'low' or contrast_level == 'high'

        if contrast_granularity == 'coarse':
            if contrast_level == 'high':
                # contrast on encoder hidden states
                seq_hidden_st = self.get_seq_hidden(
                    encoder_out_st["encoder_out"], encoder_out_st["encoder_padding_mask"][0])
                seq_hidden_mt = self.get_seq_hidden(
                    encoder_out_mt["encoder_out"], encoder_out_mt["encoder_padding_mask"][0])
            else:
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
            sim_matrix = F.cosine_similarity(
                seq_hidden_st.unsqueeze(0),
                seq_hidden_mt.unsqueeze(1),
                dim=-1
            )
            sim_matrix /= self.contrastive_temperature
            if self.use_dual_ctr:
                contrastive_loss = 0.5 * (-torch.nn.LogSoftmax(0)(sim_matrix).diag() +
                                          -torch.nn.LogSoftmax(1)(sim_matrix).diag())
            else:
                contrastive_loss = -torch.nn.LogSoftmax(0)(sim_matrix).diag()
        else:
            if contrast_level == 'high':
                audio_hidden, padding_mask_st = (encoder_out_st["encoder_out"][0],
                                                 encoder_out_st["encoder_padding_mask"][0])
                text_hidden, padding_mask_mt = (encoder_out_mt["encoder_out"][0],
                                                encoder_out_mt["encoder_padding_mask"][0])
            else:
                audio_out = model.encoder.encode_audio(
                    sample["net_input"]["src_tokens"],
                    sample["net_input"]["src_lengths"]
                )
                audio_hidden, padding_mask_st = audio_out[0], audio_out[1]
                text_hidden, padding_mask_mt = model.encoder.encode_text(sample["source"])
            text_hidden, padding_mask_mt = text_hidden[1:, :, :], padding_mask_mt[:, 2:]
            # remove eos tokens in text embedding by add a mask position
            padding_mask_mt = torch.cat(
                (
                    padding_mask_mt,
                    torch.tensor(
                        [[1]],
                        device=padding_mask_mt.device,
                        dtype=padding_mask_mt.dtype
                    ).expand(padding_mask_mt.size(0), -1)
                ),
                dim=-1
            )
            # flatten feats and padding masks
            audio_hidden = audio_hidden.transpose(0, 1).contiguous().view(-1, audio_hidden.size(-1))
            text_hidden = text_hidden.transpose(0, 1).contiguous().view(-1, text_hidden.size(-1))
            padding_mask_st = padding_mask_st.contiguous().view(-1)
            padding_mask_mt = padding_mask_mt.contiguous().view(-1)

            audio_indices = torch.nonzero((~padding_mask_st).int(), as_tuple=True)
            text_indices = torch.nonzero((~padding_mask_mt).int(), as_tuple=True)
            audio_hidden = audio_hidden.index_select(0, audio_indices[0])
            text_hidden = text_hidden.index_select(0, text_indices[0])
            assert audio_hidden.size(0) == text_hidden.size(0)
            sim_matrix = F.cosine_similarity(
                audio_hidden.unsqueeze(0),
                text_hidden.unsqueeze(1),
                dim=-1
            )
            sim_matrix /= self.contrastive_temperature
            if self.use_dual_ctr:
                contrastive_loss = 0.5 * (-torch.nn.LogSoftmax(0)(sim_matrix).diag() +
                                          -torch.nn.LogSoftmax(1)(sim_matrix).diag())
            else:
                contrastive_loss = -torch.nn.LogSoftmax(0)(sim_matrix).diag()

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
