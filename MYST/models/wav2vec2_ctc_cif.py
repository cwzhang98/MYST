# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

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
    register_model_architecture
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES, LAYER_TYPE_CHOICES, AdapterFast

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
    # mask_min_space: Optional[int] = field(
    #     default=1,
    #     metadata={"help": "min space between spans (if no overlap is enabled)"},
    # )
    # require_same_masks: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "whether to number of masked timesteps must be the same across all "
    #                 "examples in a batch"
    #     },
    # )
    # mask_dropout: float = field(
    #     default=0.0,
    #     metadata={"help": "percent of masks to unmask for each sample"},
    # )

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
    # mask_channel_before: bool = False
    # normalize: bool = II("task.normalize")
    # update_alibi: bool = True
    # data: str = II("task.data")
    # this holds the loaded wav2vec args
    # w2v_args: Any = None
    # offload_activations: bool = field(
    #     default=False, metadata={"help": "offload_activations"}
    # )
    # min_params_to_wrap: int = field(
    #     default=int(1e8),
    #     metadata={
    #         "help": "minimum number of params for a layer to be wrapped with FSDP() when "
    #                 "training with --ddp-backend=fully_sharded. Smaller values will "
    #                 "improve memory efficiency, but may make torch.distributed "
    #                 "communication less efficient due to smaller input sizes. This option "
    #                 "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
    #                 "--offload-activations are passed."
    #     },
    # )

    # checkpoint_activations: bool = field(
    #     default=False,
    #     metadata={"help": "recompute activations and save memory for extra compute"},
    # )
    # ddp_backend: str = II("distributed_training.ddp_backend")

    # zero_mask: bool = False

    # layer_decay: float = 1

    # layer_type: LAYER_TYPE_CHOICES = field(
    #     default="transformer", metadata={"help": "layer type in encoder"}
    # )
    # Adapter num
    # adp_num: int = field(
    #     default=-1
    # )
    # adp_dim: int = field(
    #     default=64
    # )
    # adp_act_fn: str = field(
    #     default="relu"
    # )
    # adp_trf_idx: str = field(
    #     default="all",
    # )
    #
    # freeze_regex: Optional[str] = field(
    #     default=None,
    # )


def add_wav2vec_args(parser):
    parser.add_argument("--pt-type", choices=["ctc", "cif", "joint"])
    parser.add_argument("--w2v-path", type=str, help="path to wav2vec 2.0 model")
    parser.add_argument("--dropout", type=float, help="dropout probability inside wav2vec 2.0 model")
    parser.add_argument("--normalized", type=bool, action="store_true", default=False)
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        default=False,
        help="if true, does not load pretrained weights",
    )
    parser.add_argument(
        "--dropout-input",
        type=float,
        metavar="D",
        help="dropout to apply to the input (after feat extr)",
    )
    parser.add_argument(
        "--final-dropout",
        type=float,
        metavar="D",
        help="dropout after transformer and before final projection",
    )
    parser.add_argument(
        "--apply-mask", action="store_true", help="apply masking during fine-tuning"
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--activation-dropout",
        "--relu-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside wav2vec 2.0 model",
    )

    parser.add_argument(
        "--mask-length", type=int, help="repeat the mask indices multiple times"
    )
    parser.add_argument(
        "--mask-prob", type=float, help="probability of replacing a token with mask"
    )

    parser.add_argument(
        "--mask-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )
    parser.add_argument(
        "--mask-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--mask-channel-length", type=int, help="repeat the mask indices multiple times"
    )
    parser.add_argument(
        "--mask-channel-prob",
        type=float,
        help="probability of replacing a token with mask",
    )

    parser.add_argument(
        "--mask-channel-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-channel-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-channel-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--freeze-finetune-updates",
        default=0,
        type=int,
        help="dont finetune wav2vec for this many updates",
    )
    parser.add_argument(
        "--feature-grad-mult",
        default=None,
        type=float,
        help="reset feature grad mult in wav2vec 2.0 to this",
    )

    parser.add_argument(
        "--layerdrop",
        default=0.0,
        type=float,
        help="probability of dropping a layer in wav2vec 2.0",
    )


# @dataclass
# class Wav2Vec2CtcConfig(Wav2Vec2AsrConfig):
#     blank_weight: float = 0
#     blank_mode: str = "add"


@register_model("wav2vec_ctc_cif")
class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, args, w2v_encoder):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args  # command-line args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_wav2vec_args(parser)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args, task.target_dictionary)
        return cls(args, w2v_encoder)

    @staticmethod
    def get_logits(net_output, normalize=False):
        logits = net_output["encoder_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            # logits: T x B x C  padding_mask: B x T
            if logits.size(0) > net_output["padding_mask"].size(1):
                # extend padding matrix to B x (1 + T)
                net_output["padding_mask"] = F.pad(
                    net_output["padding_mask"], (1, 0), value=False
                )
            # mask encoder_out(set pad feature with masking_tensor)
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, args, tgt_dict=None):
        self.apply_mask = args.apply_mask

        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }

        if args.w2v_args is None:
            # load and process checkpoint, overrides above args to checkpoint args
            state = checkpoint_utils.load_checkpoint_to_cpu(args.w2v_path, arg_overrides)
            # state['cfg'] ==  state['args']
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None

            args.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = args.w2v_args
            if isinstance(w2v_args, Namespace):
                args.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        print(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert args.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = args.data
        task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
        model = task.build_model(w2v_args.model, from_checkpoint=True)
        # remove modules won't be used in fine-tune
        model.remove_pretraining_modules()
        d = w2v_args.model.encoder_embed_dim

        # load state dict
        if state is not None and not args.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        super().__init__(task.source_dictionary)

        self.w2v_model = model

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0

        self.proj = None

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(args, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, args.decoder_embed_dim)

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

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
        }

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


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("wav2vec_ctc_cif", "wav2vec_ctc_cif_base")
def base_architecture(args):
    args.w2v_path = getattr(args, "w2v_path", "./checkpoints/wav2vec_small.pt")
