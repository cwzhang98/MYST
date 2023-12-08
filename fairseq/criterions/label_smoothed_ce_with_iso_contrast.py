import math
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from typing import List, Optional

from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion,
    label_smoothed_nll_loss
)
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyWithIsoConstrastConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    contrastive_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of contrastive loss"}
    )
    muti_contrast: Optional[bool] = field(
        default=False,
        metadata={"help": "muti grained contrastive loss"}
    )
    contrast_granularity: Optional[str] = field(
        default=None,
        metadata={
            "help": "if `muti_contrast` set to False, assign granularity of single loss, could be `fine` or `coarse`"}
    )
    use_muti_layer_repr_for_contrast: Optional[bool] = field(
        default=False,
        metadata={"help": "refer to EMNLP 2020: On the Sentence Embeddings from Pre-trained Language Models;"
                          "default to use representation of the first Layer and the last layer"}
    )
    contrastive_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "the temperature in the contrastive loss"}
    )
    use_dual_ctr: Optional[bool] = field(
        default=False,
        metadata={"help": "if we want to use dual contrastive loss"}
    )
    iso_transform: Optional[str] = field(
        default=None,
        metadata={"help": "appaoach to transform aniso distributed repr to iso"}
    )
    ablation_type: Optional[str] = II("model.ablation_type")


@register_criterion(
    "label_smoothed_cross_entropy_with_iso_contrast",
    dataclass=LabelSmoothedCrossEntropyWithIsoConstrastConfig
)
class LabelSmoothedCrossEntropyWithIsoConstrast(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            contrastive_weight=0.0,
            muti_contrast=False,
            contrast_granularity=None,
            use_muti_layer_repr_for_contrast=False,
            contrastive_temperature=1.0,
            use_dual_ctr=False,
            iso_transform=None,
            ablation_type=None
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.contrastive_weight = contrastive_weight
        self.muti_contrast = muti_contrast
        self.contrast_granularity = contrast_granularity
        self.use_muti_layer_repr_for_contrast = use_muti_layer_repr_for_contrast
        self.contrastive_temperature = contrastive_temperature
        self.use_dual_ctr = use_dual_ctr
        self.iso_transform = iso_transform
        self.ablation_type = ablation_type

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], is_audio_input=True)

        contrastive_loss = torch.tensor(0.0)
        fine_contrast_loss, coarse_contrast_loss = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss, nll_loss = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_mt, nll_loss_mt = torch.tensor(0.0), torch.tensor(0.0)
        jsd_loss, qua_loss = torch.tensor(0.0), torch.tensor(0.0)

        if model.training:
            decoder_out, encoder_out = net_output  # encoder output include audio and textual encoder
        else:
            decoder_out = net_output
        # muti task loss
        if sample["target"] is not None:
            # st cross entropy
            label_smoothed_nll_loss, nll_loss, lprobs_st, probs_st, target = self.compute_loss(
                model, decoder_out, sample, reduce=reduce
            )
        if model.training and self.ablation_type != "w2v_transformer":
            # mt cross entropy
            label_smoothed_nll_loss_mt, nll_loss_mt, lprobs_mt, probs_mt, \
                text_embedding, text_padding_mask, text_encoder_states = self.compute_loss_mt(
                model, sample, reduce=True
            )
            # qua loss
            if sample["net_input"]["transcript_lengths"] is not None:
                alpha_sum = torch.sum(encoder_out["alpha"], dim=-1) \
                            / model.encoder.w2v_cif_model.beta
                source_lengths = sample["source_lengths"].type_as(alpha_sum)
                qua_loss = F.l1_loss(alpha_sum, source_lengths, reduction='sum')
            # jsd loss
            lprobs_mix = 0.5 * (lprobs_st + lprobs_mt)
            jsd_loss = self.compute_loss_jsd(lprobs_mix, probs_st, probs_mt, target, reduce=reduce)

            # contrastive loss
            audio_embedding = encoder_out["encoder_embedding"]
            audio_padding_mask = encoder_out["encoder_padding_mask"][0]
            audio_encoder_states = encoder_out["shared_encoder_states"]
            if self.muti_contrast:
                fine_contrast_loss, coarse_contrast_loss = self.compute_contrastive_loss(
                    audio_embedding,
                    audio_padding_mask,
                    audio_encoder_states,
                    text_embedding,
                    text_padding_mask,
                    text_encoder_states,
                    reduce=reduce,
                    muti_contrast=True
                )
                contrastive_loss = fine_contrast_loss + coarse_contrast_loss
            else:
                contrastive_loss = self.compute_contrastive_loss(
                    audio_embedding,
                    audio_padding_mask,
                    audio_encoder_states,
                    text_embedding,
                    text_padding_mask,
                    text_encoder_states,
                    reduce=reduce,
                    muti_contrast=False,
                    contrast_granularity=self.contrast_granularity
                )
        else:
            contrastive_loss = torch.tensor(0.0)

        muti_task_ce_loss = label_smoothed_nll_loss + label_smoothed_nll_loss_mt
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]

        loss = muti_task_ce_loss + jsd_loss + self.contrastive_weight * contrastive_loss + qua_loss

        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),  # st nll loss
            "nll_loss_mt": utils.item(nll_loss_mt.data),
            "jsd_loss": utils.item(jsd_loss.data),  # 0, if not training
            "qua_loss": utils.item(qua_loss.data),  # 0, if not training
            "contrastive_loss": utils.item(contrastive_loss.data),
            "fine_contrast_loss": utils.item(fine_contrast_loss.data),
            "coarse_contrast_loss": utils.item(coarse_contrast_loss.data),
            "ntokens": sample["target_ntokens"],
            "source_ntokens": sample["source_ntokens"],  # num of mt src token in a batch
            "target_tokens": sample["target_ntokens"],  # num of tgt token in a batch
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, decoder_out, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        target = model.get_targets(sample, net_output).long()
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            probs = probs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        lprobs = probs.log()
        return lprobs.view(-1, lprobs.size(-1)), probs.view(-1, probs.size(-1)), target.view(-1)

    def compute_accuracy(self, model, net_output, sample):
        lprobs, _, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, probs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, probs, target

    def compute_loss_mt(self, model, sample, reduce=True):
        decoder_out, encoder_out = model(
            sample["source"],
            sample["source_lengths"],
            sample["net_input"]["prev_output_tokens"],
            is_audio_input=False
        )
        text_embedding = encoder_out["encoder_embedding"]
        text_padding_mask = encoder_out["encoder_padding_mask"][0]
        text_encoder_states = encoder_out["shared_encoder_states"]
        probs = model.get_normalized_probs(decoder_out, log_probs=False)
        target = model.get_targets(sample, decoder_out)
        if self.ignore_prefix_size > 0:
            probs = probs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        lprobs = probs.log()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        probs = probs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce
        )
        return loss, nll_loss, lprobs, probs, text_embedding, \
            text_padding_mask, text_encoder_states

    def compute_loss_jsd(
            self, lprobs_mix, probs_st, probs_mt, target, reduce=True
    ):
        kl_st = F.kl_div(lprobs_mix, probs_st, reduction="none").sum(-1)
        kl_mt = F.kl_div(lprobs_mix, probs_mt, reduction="none").sum(-1)
        pad_mask = target.eq(self.padding_idx)
        kl_mt.masked_fill_(pad_mask, 0.0)
        kl_st.masked_fill_(pad_mask, 0.0)
        if reduce:
            jsd_loss = 0.5 * (kl_mt.sum() + kl_st.sum())
        else:
            jsd_loss = 0.5 * (kl_mt + kl_st)
        return jsd_loss

    def get_sentence_repr(self, encoder_states_list: List, padding_mask):
        """get sentence level from hidden states"""
        padding_mask = (~padding_mask)
        B, C = encoder_states_list[0].size(1), encoder_states_list[0].size(2)
        if self.use_muti_layer_repr_for_contrast:
            assert len(encoder_states_list) > 1, \
                "mutipule layer hidden states are required"
            seq_hidden = encoder_states_list[0].new_zeros((B, C))
            for states in encoder_states_list:
                states = states.transpose(0, 1)
                seq_hidden += (states * padding_mask.unsqueeze(-1)).sum(1) \
                              / padding_mask.sum(1).unsqueeze(-1)  # B x C
            seq_hidden /= len(encoder_states_list)
        else:
            assert len(encoder_states_list) == 1
            seq_hidden = (encoder_states_list[0].transpose(0, 1) * padding_mask.unsqueeze(-1)).sum(1) \
                         / padding_mask.sum(1).unsqueeze(-1)
        return seq_hidden

    def compute_contrastive_loss(
            self,
            audio_embedding,  # T x B x C
            audio_padding_mask,  # B x T
            audio_encoder_states,  # T x B x C
            text_embedding,  # T x B x C
            text_padding_mask,  # B x T
            text_encoder_states,
            reduce=True,
            muti_contrast=False,
            contrast_granularity=None
    ):
        fine_loss, coarse_loss = torch.tensor(0.0), torch.tensor(0.0)
        if muti_contrast or contrast_granularity == "fine":  # word level loss
            # remove lang tag feature
            text_embedding, _text_padding_mask = text_embedding[1:, :, :], text_padding_mask[:, 2:]
            # remove eos tokens in text embedding by add an additional mask position
            _text_padding_mask = torch.cat(
                (
                    _text_padding_mask,
                    torch.tensor(
                        [[1]],
                        device=_text_padding_mask.device,
                        dtype=_text_padding_mask.dtype
                    ).expand(_text_padding_mask.size(0), -1)
                ),
                dim=-1
            )
            # flatten feats and padding masks
            audio_embedding = audio_embedding.transpose(0, 1) \
                .contiguous() \
                .view(-1, audio_embedding.size(-1))
            text_embedding = text_embedding.transpose(0, 1) \
                .contiguous() \
                .view(-1, text_embedding.size(-1))
            audio_padding_mask_flat = audio_padding_mask.contiguous().view(-1)
            text_padding_mask_flat = _text_padding_mask.contiguous().view(-1)
            # selet feats according to mask
            audio_index = torch.nonzero((~audio_padding_mask_flat).int(), as_tuple=True)
            text_index = torch.nonzero((~text_padding_mask_flat).int(), as_tuple=True)
            audio_feats = audio_embedding.index_select(0, audio_index[0])  # (B x T) x C
            text_feats = text_embedding.index_select(0, text_index[0])
            # audio feats length equals to text feats length minus 2
            assert audio_feats.size(0) == text_feats.size(0)
            feats_length, hidden_size = audio_feats.size()
            sim_matrix = F.cosine_similarity(
                audio_feats.expand(
                    feats_length, feats_length, hidden_size),
                text_feats.expand(
                    feats_length, feats_length, hidden_size).transpose(0, 1),
                dim=-1
            )
            sim_matrix /= self.contrastive_temperature
            if self.use_dual_ctr:
                fine_loss = 0.5 * (
                        -torch.nn.LogSoftmax(0)(sim_matrix).diag() + \
                        -torch.nn.LogSoftmax(1)(sim_matrix).diag()
                )
            else:
                fine_loss = -torch.nn.LogSoftmax(0)(sim_matrix).diag()
            if reduce:
                fine_loss = fine_loss.sum()

        if muti_contrast or contrast_granularity == "coarse":  # sentence level loss
            audio_sentence_repr = self.get_sentence_repr(audio_encoder_states, audio_padding_mask)
            text_sentence_repr = self.get_sentence_repr(text_encoder_states, text_padding_mask)
            batch_size, hidden_size = audio_sentence_repr.size()
            sim_matrix = F.cosine_similarity(
                audio_sentence_repr.expand(batch_size, batch_size, hidden_size),
                text_sentence_repr.expand(batch_size, batch_size, hidden_size).transpose(0, 1),
                dim=-1
            )
            sim_matrix /= self.contrastive_temperature
            if self.use_dual_ctr:
                coarse_loss = 0.5 * (
                        -torch.nn.LogSoftmax(0)(sim_matrix).diag() + \
                        -torch.nn.LogSoftmax(1)(sim_matrix).diag()
                )
            else:
                coarse_loss = -torch.nn.LogSoftmax(0)(sim_matrix).diag()
            if reduce:
                coarse_loss = coarse_loss.sum()
        if not muti_contrast and contrast_granularity not in {"fine", "coarse"}:
            raise Exception("if muti_contrast set to False, "
                            "contrast_granularity should be `fine` or `coarse`")
        if muti_contrast:
            return fine_loss, coarse_loss
        else:
            return fine_loss if contrast_granularity == "fine" else coarse_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_st_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        nll_loss_mt_sum = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        fine_contrast_loss_sum = sum(log.get("fine_contrast_loss", 0) for log in logging_outputs)
        coarse_contrast_loss_sum = sum(log.get("coarse_contrast_loss", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)
        qua_loss_sum = sum(log.get("qua_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_st",
                           nll_loss_st_sum / sample_size / math.log(2), sample_size, round=3)
        if nll_loss_mt_sum > 0:
            metrics.log_scalar("nll_loss_mt",
                               nll_loss_mt_sum / sample_size / math.log(2), sample_size, round=3)
        if contrastive_loss_sum > 0:
            metrics.log_scalar("contrasitve_loss",
                               contrastive_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if jsd_loss_sum > 0:
            metrics.log_scalar("jsd_loss",
                               jsd_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if qua_loss_sum > 0:
            metrics.log_scalar("qua_loss",
                               qua_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if fine_contrast_loss_sum > 0:
            metrics.log_scalar("fine_contrast_loss",
                               fine_contrast_loss_sum / sample_size / math.log(2),
                               sample_size, round=3)
        if coarse_contrast_loss_sum > 0:
            metrics.log_scalar("coarse_contrast_loss",
                               coarse_contrast_loss_sum / sample_size / math.log(2),
                               sample_size, round=3)

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
