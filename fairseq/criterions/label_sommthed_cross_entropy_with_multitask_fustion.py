import math
import torch
from dataclasses import dataclass, field
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss
)
from fairseq.criterions.label_smoothed_cross_entropy_with_multitask import (
    LabelSmoothedCrossEntropyWithMultitask
)


@dataclass
class LabelSmoothedCrossEntropyWithMultitaskFusionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    fusion_disable_epochs: int = field(
        default=0,
        metadata={"help": "train several epochs before fuse features"}
    )
    reconstruction_weight: float = field(
        default=1.0,
        metadata={"help": "weight of reconstruction loss"}
    )
    fusion_weight: float = field(
        default=1.0,
        metadata={"help": "weight of fusion loss"}
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_multitask_fusion",
    dataclass=LabelSmoothedCrossEntropyWithMultitaskFusionConfig
)
class LabelSmoothedCrossEntropyWithMultitaskFusion(
    LabelSmoothedCrossEntropyWithMultitask
):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            fusion_disable_steps=0,
            reconstruction_weight=1.0,
            fusion_weight=1.0
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.fusion_disable_steps = fusion_disable_steps
        self.reconstruction_weight = reconstruction_weight
        self.fusion_weight = fusion_weight
        self.pad_idx = task.tgt_dict.pad()

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], is_audio_input=True)
        st_loss, nll_loss_st = torch.tensor(0.), torch.tensor(0.)
        mt_loss, nll_loss_mt = torch.tensor(0.), torch.tensor(0.)
        ft_loss, nll_loss_ft = torch.tensor(0.), torch.tensor(0.)
        ctc_loss, reconstruction_loss = torch.tensor(0.), torch.tensor(0.)
        fusion_loss, jsd_loss_st, jsd_loss_ft = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        jsd_loss_st_rdrop = torch.tensor(0.)
        if model.training:
            net_output, encoder_out = net_output
        if sample["target"] is not None:
            # forward st
            st_loss, nll_loss_st, lprobs_st = self.compute_loss(
                model, net_output, sample, reduce=reduce)
        if model.training:
            # forward mt
            mt_loss, nll_loss_mt, lprobs_mt = self.compute_loss_mt(
                model, sample, reduce=reduce)
            _, ctc_padding_mask, ctc_lprobs, audio_feats = model.encoder.compute_ctc_logits_and_lprobs(
                sample['net_input']['src_tokens'],
                sample['net_input']['src_lengths']
            )
            ctc_loss = self.compute_loss_ctc(
                sample,
                ctc_lprobs,
                ctc_padding_mask,
                reduce=reduce
            )
            audio_embed = model.encoder.position_embed(audio_feats, ctc_padding_mask)
            text_feats, text_padding_mask = model.encoder.encode_text(sample["source"])
            concat_feats = torch.cat((audio_embed, text_feats), dim=0)
            concat_padding_mask = torch.cat((encoder_out['encoder_padding_mask'][0], text_padding_mask), dim=-1)
            # forward concat feats in TEnc
            concat_encoder_out = model.encoder.forward_shared_encoder(concat_feats, concat_padding_mask)
            # forward concat feats in TDec
            audio_fused_out = {
                "encoder_out": [concat_encoder_out['encoder_out'][0][:audio_feats.size(0), :, :]],  # T x B x C
                "encoder_padding_mask": encoder_out['encoder_padding_mask']
            }
            concat_decoder_out = model.decoder(
                prev_output_tokens=sample['net_input']['prev_output_tokens'],
                encoder_out=audio_fused_out
            )
            # apply mask
            if model.num_updates > 20000:
                masked_audio_feats, ctc_results, audio_mask_indices = model.encoder.mask_audio_feats(
                    audio_feats,
                    encoder_out['encoder_padding_mask'][0],
                    ctc_lprobs,
                    self.blank_idx
                )
                concat_feats_masked = torch.cat((masked_audio_feats, text_feats), dim=0)
                # forward masked concat feats in TEnc
                concat_encoder_out_masked = model.encoder.forward_shared_encoder(concat_feats_masked,
                                                                                 concat_padding_mask)
                # loss
                reconstruction_lprobs = model.encoder.compute_reconstruction_lprobs(
                    concat_encoder_out_masked['encoder_out'][0][:masked_audio_feats.size(0), :, :])
                reconstruction_loss = self.label_smoothed_mlm_loss(
                    reconstruction_lprobs.transpose(0, 1), ctc_results, self.eps, audio_mask_indices
                )
            # reconstruction_loss = self.compute_loss_ctc(
            #     sample,
            #     reconstruction_lprobs,
            #     encoder_out['encoder_padding_mask'][0],
            #     reduce=reduce
            # )
            # fusion_loss = torch.nn.functional.mse_loss(
            #     audio_fused_out['encoder_out'][0].float(), encoder_out['encoder_out'][0].float(), reduction='sum'
            # )
            ft_loss, nll_loss_ft, lprobs_ft = self.compute_loss(model, concat_decoder_out, sample, reduce)
            jsd_loss_st = self.compute_loss_jsd(lprobs_st, lprobs_mt.detach(), sample['target'])
            jsd_loss_ft = self.compute_loss_jsd(lprobs_ft, lprobs_mt.detach(), sample['target'])
            jsd_loss_st_rdrop = self.compute_loss_jsd(lprobs_ft, lprobs_st, sample['target'])

        loss = (st_loss + mt_loss + ctc_loss + ft_loss + self.reconstruction_weight * reconstruction_loss +
                self.fusion_weight * fusion_loss + jsd_loss_st + jsd_loss_ft + jsd_loss_st_rdrop)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["target_ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss_st": utils.item(nll_loss_st.data),
            "nll_loss_mt": utils.item(nll_loss_mt.data),
            "nll_loss_ft": utils.item(nll_loss_ft.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "reconstruction_loss": utils.item(reconstruction_loss.data),
            "fusion_loss": utils.item(fusion_loss.data),
            "jsd_loss_st": utils.item(jsd_loss_st.data),
            "jsd_loss_ft": utils.item(jsd_loss_ft.data),
            "jsd_loss_st_rdrop": utils.item(jsd_loss_st_rdrop.data),
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

    def label_smoothed_mlm_loss(self, lprobs, target, epsilon, mask_indices):
        """
            mask_indices: B x T tensor of masked indices
        """
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)  # 0.1/9999
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        loss[~mask_indices] = 0.0
        return loss.sum()

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_st_sum = sum(log.get("nll_loss_st", 0) for log in logging_outputs)
        nll_loss_mt_sum = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)
        nll_loss_ft_sum = sum(log.get("nll_loss_ft", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        reconstruction_loss_sum = sum(log.get("reconstruction_loss", 0) for log in logging_outputs)
        fusion_loss_sum = sum(log.get("fusion_loss", 0) for log in logging_outputs)
        jsd_loss_ft_sum = sum(log.get("jsd_loss_ft", 0) for log in logging_outputs)
        jsd_loss_st_sum = sum(log.get("jsd_loss_st", 0) for log in logging_outputs)
        jsd_loss_st_rdrop_sum = sum(log.get("jsd_loss_st_rdrop", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)


        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_st",
                           nll_loss_st_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_mt",
                           nll_loss_mt_sum / sample_size / math.log(2), sample_size, round=3)
        if nll_loss_ft_sum > 0:
            metrics.log_scalar("nll_loss_ft",
                               nll_loss_ft_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("ctc_loss",
                           ctc_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if reconstruction_loss_sum > 0:
            metrics.log_scalar("reconstruction_loss",
                               reconstruction_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if fusion_loss_sum > 0:
            metrics.log_scalar("fusion_loss",
                               fusion_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("jsd_loss_st",
                           jsd_loss_st_sum / sample_size / math.log(2), sample_size, round=3)
        if jsd_loss_ft_sum > 0:
            metrics.log_scalar("jsd_loss_ft",
                               jsd_loss_ft_sum / sample_size / math.log(2), sample_size, round=3)
        if jsd_loss_st_rdrop_sum > 0:
            metrics.log_scalar("jsd_loss_st_rdrop",
                               jsd_loss_st_rdrop_sum / sample_size / math.log(2), sample_size, round=3)
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