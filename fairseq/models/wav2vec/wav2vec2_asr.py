# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import math
import re
from omegaconf import OmegaConf
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    register_model,
)
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES, LAYER_TYPE_CHOICES
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.tasks import FairseqTask

from fairseq.models.wav2vec.cif import cif_function
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class Wav2Vec2AsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )

    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    drop_path: float = 0
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    normalize: bool = II("task.normalize")
    update_alibi: bool = True
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None
    ddp_backend: str = II("distributed_training.ddp_backend")

    zero_mask: bool = False
    load_ema: bool = False

    layer_decay: float = 1


    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )

    freeze_regex: Optional[str] = field(
        default=None,
    )
    # cif config
    weight_predictor_stop_gradient: bool = field(
        default=True,
        metadata={"help": "stop backward pass from weight predictor to lower arch"}
    )
    use_attention: bool = field(
        default=False,
        metadata={"help": "use self-attention in weight predictor"}
    )
    cif_attention_dropout: float = field(
        default=0.0,
        metadata={"help": "attention dropout used in weight predictor"}
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "fire threshold value"}
    )
    tail_threshold: float = field(
        default=0.5,
        metadata={"help": "threshold to decide whether to perform tail handling"}
    )
    ctc_joint_training: bool = field(
        default=False,
        metadata={"help": "ctc joint training when perform cif"}
    )
    shared_proj: bool = field(
        default=False,
        metadata={"help": "shared weight between ctc and cif proj"}
    )
    sub_sampler:bool = field(
        default=False,
        metadata={"help": "sub_sampler after wav2vec"}
    )
    conv_channels: int = field(
        default=1024,
        metadata={"help": "intermediate channels"}
    )
    textual_encoder_embed_dim: int = 512
    conv_kernel_sizes: str = "5,5"

@register_model("wav2vec_ctc", dataclass=Wav2Vec2AsrConfig)
class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2AsrConfig, w2v_encoder: BaseFairseqModel, target_dim):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.hidden_dim = cfg.w2v_args.model.encoder_embed_dim
        self.ctc_joint_training = cfg.ctc_joint_training
        self.beta = cfg.beta
        self.cif_proj = None
        self.subsample_audio = None
        self.textual_dim_proj = None
        # ctc proj
        if self.ctc_joint_training:
            self.ctc_proj = Linear(cfg.textual_encoder_embed_dim, target_dim)
        if not cfg.shared_proj or not self.ctc_joint_training:
            self.cif_proj = Linear(cfg.textual_encoder_embed_dim, target_dim)
        
        if cfg.sub_sampler:
            self.subsample_audio = Conv1dSubsampler(
                self.hidden_dim,  # 768
                cfg.conv_channels,  # 1024
                cfg.textual_encoder_embed_dim,  # 512
                [int(k) for k in cfg.conv_kernel_sizes.split(",")]
            )
        else:
            self.textual_dim_proj = nn.Sequential(
                Linear(self.hidden_dim, 1024),
                nn.Dropout(p=cfg.dropout, inplace=True),
                nn.GELU(),
                Linear(1024, cfg.textual_encoder_embed_dim),
                nn.Dropout(p=cfg.dropout, inplace=True),
                LayerNorm(cfg.textual_encoder_embed_dim),
            )
        self._build_cif(cfg)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2AsrConfig, task_or_dict):
        """Build a new model instance."""
        if isinstance(cfg, dict):
            cfg = Wav2Vec2AsrConfig(**cfg)
        w2v_encoder = Wav2VecEncoder(cfg)
        if isinstance(task_or_dict, Dictionary):  # st ft
            return cls(cfg, w2v_encoder, len(task_or_dict))
        else:  # cif pt
            return cls(cfg, w2v_encoder, len(task_or_dict.target_dictionary))
    
    def _build_cif(self, cfg):
        self.cif_layer = CIFLayer(cfg)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"] # (B, T, C)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_ctc_output(self, net_output):
        logits = net_output["ctc_proj_out"] # (B, T, C)
        lprobs = utils.log_softmax(logits.float(), dim=-1)
        lens = lprobs.new_full((lprobs.shape[0],), lprobs.shape[1]).long()
        if len(net_output["padding_mask"]) > 0:
            lens -= net_output["padding_mask"].sum(dim=-1)
        return lprobs.transpose(1, 0), lens

    def get_ctc_target(self, sample):
        return sample["target"].long(), sample["target_lengths"].long()

    def extract_features(self, transcript_lengths, source, padding_mask, freeze_w2v):
        """
            similar to forward but only extract acoustic features
        """
        if freeze_w2v:
            with torch.no_grad():
                x = self.w2v_encoder(source, padding_mask, mask=False)
        else:
            x = self.w2v_encoder(source, padding_mask, mask=False)
        
        if self.subsample_audio is not None:
            output_length = (1 - x["padding_mask"].int()).sum(dim=1)
            feats, output_length = self.subsample_audio(x["encoder_out"], output_length)
            x["encoder_out"] = feats.transpose(1, 0).contiguous() 
            x["padding_mask"] = lengths_to_padding_mask(output_length)
        else:
            x["encoder_out"] = self.textual_dim_proj(x["encoder_out"]) # 768 -> 512
        
        if self.training:
            assert transcript_lengths is not None
            
        cif_out = self.cif_layer(
            x["encoder_out"],
            x["padding_mask"],
            transcript_lengths
        )
        return cif_out["cif_out"][0], cif_out["cif_length"][0], cif_out["alpha"][0] # cif aggraved features

    def remove_pretrain_modules(self):
        self.ctc_proj = None
        self.cif_proj = None
    
    def forward(self, transcript_lengths, source, padding_mask, **kwargs):
        x = self.w2v_encoder(source, padding_mask, **kwargs)

        if self.subsample_audio is not None:
            output_length = (1 - x["padding_mask"].int()).sum(dim=1)
            feats, output_length = self.subsample_audio(x["encoder_out"], output_length)
            x["encoder_out"] = feats.transpose(1, 0).contiguous() 
            x["padding_mask"] = lengths_to_padding_mask(output_length)
        else:
            x["encoder_out"] = self.textual_dim_proj(x["encoder_out"])  # 768 -> 512

        if self.ctc_joint_training:
            ctc_proj_out = self.ctc_proj(x["encoder_out"])

        if self.training:
            assert transcript_lengths is not None
        cif_out = self.cif_layer(
            x["encoder_out"],
            x["padding_mask"],
            transcript_lengths
        )
        
        if self.cif_proj: # not share parameters
            cif_proj_out = self.cif_proj(cif_out["cif_out"][0])
        else:
            cif_proj_out = self.ctc_proj(cif_out["cif_out"][0])
        x.update({
            "encoder_out": cif_proj_out,  # cif logits; B x T x C
            "ctc_proj_out": ctc_proj_out,  # ctc logits; B x T x C
            "alpha": cif_out["alpha"][0],
            "cif_length": cif_out["cif_length"][0],
            "beta": self.beta
        })
        return x


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:  # in pt, args should be none
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args
        else:  # in ST fine tune, pass the checkpoint args
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, dict):
                cfg.w2v_args = w2v_args = OmegaConf.create(w2v_args)
            if isinstance(w2v_args, Namespace): # cfg should be instanced of Namespace
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
        model = task.build_model(w2v_args.model, from_checkpoint=True)
        model.remove_pretraining_modules()
        # load checkpoint
        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We don't load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            to_delete = {"_ema", "target_proj", "decoder"}
            for k in to_delete:
                if k in state["model"]:
                    del state["model"][k]

            if hasattr(model, "modality_encoders"):
                if "modality_encoders.AUDIO.encoder_mask" not in state["model"]:
                    model.modality_encoders["AUDIO"].encoder_mask = None
                elif not cfg.zero_mask:
                    model.modality_encoders["AUDIO"].encoder_mask = None
                    del state["model"]["modality_encoders.AUDIO.encoder_mask"]

                for k in list(state["model"].keys()):
                    if k.startswith("modality_encoders.") and not k.startswith(
                        "modality_encoders.AUDIO"
                    ):
                        del state["model"][k]

            print(model)
            model.load_state_dict(state["model"], strict=True)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }
        if "corpus_key" in kwargs:
            w2v_args["corpus_key"] = kwargs["corpus_key"]
        if "mask" in kwargs:
            w2v_args["mask"] = kwargs["mask"]

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x, padding_mask = res
            if padding_mask == None:
                padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=bool, device=x.device)
            # else:
            #     padding_mask = res["padding_mask"]

        x = self.final_dropout(x)

        return {
            "encoder_out": x,  # B x T x C
            "padding_mask": padding_mask,  # B x T,
            #"layer_results": res["layer_results"],
        }

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class CIFLayer(nn.Module):
    def __init__(self, cfg: Wav2Vec2AsrConfig):
        super().__init__()
        self.cfg = cfg
        # initalize weight predictor
        hidden_dim = cfg.textual_encoder_embed_dim
        self.weight_predictor = WeightPredictor(
            hidden_dim,
            cfg.use_attention,
            cfg.cif_attention_dropout
        )
        self.beta = cfg.beta
        self.tail_threshold = cfg.tail_threshold
    
    def forward(
        self,
        x: Tensor,
        encoder_padding_mask: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None
    ):
        if self.cfg.weight_predictor_stop_gradient:
            alpha = self.weight_predictor(x.detach(), encoder_padding_mask)
        else:
            alpha = self.weight_predictor(x, encoder_padding_mask)
        
        alpha = alpha.squeeze(-1) # (B, T, 1) -> # (B, T)
        
        # apply mask
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.unsqueeze(2), 0)
            alpha = alpha.masked_fill(encoder_padding_mask, 0)
        # apply cif
        cif_out = cif_function(
            input=x,
            alpha=alpha,
            beta=self.beta,
            tail_threshold=self.tail_threshold,
            target_lengths=target_lengths
        )
        cif_feats = cif_out["cif_out"][0]
        cif_out.update({
            "cif_out": [cif_feats], # (B, T, C)
            "alpha": [alpha]
        })
        return cif_out


class WeightPredictor(nn.Module):
    def __init__(
            self,
            hidden_dim,
            use_attention,
            attention_dropout
        ):
        super().__init__()
        self.use_attention = use_attention
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding="same"
        )
        self.layer_norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.proj = Linear(hidden_dim, 1)
        self.gelu = nn.GELU()
        if self.use_attention:
            self.self_attention = MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                kdim=hidden_dim,
                vdim=hidden_dim,
                dropout=attention_dropout,
                self_attention=True
            )
    
    def forward(self, x: Tensor, encoder_padding_mask):
        x = x.transpose(1, 2).contiguous() # B x T x C   -> B x C x T
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(1, 0).contiguous() # B x C x T   -> T x B x C
        # pre norm
        if self.use_attention:
            residual = x
        x = self.layer_norm(x)
        if self.use_attention:    
            x, _ = self.self_attention(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = x + residual
        x = self.gelu(x)
        x = self.dropout(x)

        return self.proj(x).sigmoid().transpose(1, 0)
        

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class Conv1dSubsampler(nn.Module):

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)
