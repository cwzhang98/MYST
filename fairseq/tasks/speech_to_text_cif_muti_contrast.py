import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from omegaconf import DictConfig
import torch
from torch import Tensor
from argparse import Namespace
from fairseq.data import encoders
from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.data.dictionary import Dictionary
from fairseq.tasks import register_task, FairseqTask
from fairseq.data.audio.speech_text_triple_datatset import (
    SpeechTextTripleDataset,
    SpeechTextTripleDatasetCreator
)
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data.audio.multi_modality_dataset import ModalityDatasetItem, MultiModalityDataset
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    get_features_or_waveform
)

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@dataclass
class SpeechToTextCifMutiContrastConfig(FairseqDataclass):
    data: str = field(
        default='',
        metadata={"help": "manifest root path"}
    )
    src_dict_path: str = field(
        default=None,
        metadata={"help": "source text dict path"}
    )
    config_yaml: str = field(
        default='',
        metadata={"help": "Configuration YAML filename (absolute path)"}
    )
    max_audio_tokens: int = field(
        default=1000000,
        metadata={"help": "max batch of tokens in audio sequences"}
    )
    max_text_tokens: int = field(
        default=4000,
        metadata={"help": "max batch of tokens in text sequences"}
    )
    max_audio_positions: int = field(
        default=1000000,
        metadata={"help": "max number of tokens in the source audio sequence"}
    )
    max_source_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the source text sequence"}
    )
    max_target_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the target text sequence"}
    )
    lang_pairs: str = field(
        default=None,
        metadata={"help": "language pairs for text training, eg: `en-de`"}
    )
    lang_prefix_tok: str = field(
        default=None,
        metadata={"help": "starting token in decoder, eg: `<lang:de>`"}
    )
    external_mt_data: str = field(
        default="",
        metadata={"help": "path to the external parallel mt data, tsv file"}
    )
    text_data_sample_ratio: float = field(
        default=1.0,
        metadata={"help": "define MT data sample ratio in one batch"}
    )
    # dev BLEU config
    eval_bleu: bool = field(
        default=True,
        metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses');"
            "required if using --eval-bleu;use 'space' to disable detokenization;"
            "see fairseq.data.encoders for other options"
        }
    )
    eval_bleu_detok_args: str = field(
        default=None,
        metadata={"help": "args for building the tokenizer, if needed"}
    )
    eval_tokenized_bleu: bool = field(
        default=False,
        metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_bpe: str = field(
        default=None,
        metadata={"help": "args for building the bpe, if needed"}
    )
    eval_bleu_remove_bpe: str = field(
        default="sentencepiece",
        metadata={"help": "remove BPE before computing BLEU"}
    )
    eval_bleu_args: str = field(
        default=None,
        metadata={"help": "generation args for BLUE scoring, e.g., {'beam': 4, 'lenpen': 0.6}"}
    )
    eval_bleu_print_samples = field(
        default=True,
        metadata={"help": "print sample generations during validation"}
    )


@register_task("speech_to_text_cif_muti_contrast", dataclass=SpeechToTextCifMutiContrastConfig)
class SpeechToTextCifMutiContrast(FairseqTask):
    def __init__(self, cfg: SpeechToTextCifMutiContrastConfig, data_cfg,  src_dict, tgt_dict):
        self.cfg = cfg
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = data_cfg
        
    @classmethod
    def setup_task(cls, cfg: SpeechToTextCifMutiContrastConfig, **kwargs):
        data_cfg = S2TDataConfig(Path(cfg.data) / cfg.config_yaml)
        src_dict_path = Path(cfg.src_dict_path) / "dict.pho.txt"
        tgt_dict_path = Path(cfg.data) / cfg.lang_pairs / data_cfg.vocab_filename
        if not src_dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {src_dict_path}")
        if not tgt_dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {tgt_dict_path}")
        # add src lang tag to src dict
        assert cfg.lang_pairs is not None
        src_lang_tag  = f"<lang:{cfg.lang_pairs.split('-')[0]}>"
        src_dict = Dictionary.load(src_dict_path)
        src_dict.add_symbol(src_lang_tag)
        
        tgt_dict = Dictionary.load(tgt_dict_path)
        logger.info(f"source dict size: {len(src_dict)}; target dict size: {len(tgt_dict)}")
        if getattr(cfg, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        # set external MT dataset path
        if cfg.external_mt_data is not None:
            if not Path(cfg.external_mt_data).is_absolute():
                cfg.external_mt_data = Path(cfg.data) / cfg.external_mt_data

        return cls(cfg, data_cfg, src_dict, tgt_dict)

    
    def build_model(self, cfg: SpeechToTextCifMutiContrastConfig):
        model = super(SpeechToTextCifMutiContrast, self).build_model(cfg)
        if cfg.eval_bleu:
            assert cfg.eval_bleu_detok is not None, \
                "eval_bleu_detokenize is required id eval_bleu is true"
            detok_args = json.loads(cfg.eval_bleu_detok_args or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=cfg.eval_bleu_detok, **detok_args)
            )
            if cfg.eval_bleu_bpe is not None:
                self.bpe = self.build_bpe(cfg)
            else:
                self.bpe = None
            generation_args = json.loads(cfg.eval_bleu_args or "{}")
            self.sequence_generator = self.build_generator([model], Namespace(**generation_args))
        return model
    
    def build_criterion(self, cfg: SpeechToTextCifMutiContrastConfig):
        import criterions
        if (
            self.data_cfg.prepend_tgt_lang_tag or self.data_cfg.prepend_tgt_lang_tag
        ) and cfg.ignore_prefix_size != 1:
            raise ValueError("ignore_prefix_size should set to 1")
        
        return criterions.build_criterion(cfg, self)
    
    def build_tokenizer(self):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))
    
    def build_bpe(self):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))
    
    def build_generator(
        self,
        models,
        args, # fairseq.dataclass.configs.GenerationConfig
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None
    ):
        langs = self.cfg.lang_pairs.split("-")
        lang_token_ids = {
            self.tgt_dict.indices[SpeechTextTripleDataset.LANG_TAG_TEMPLATE.format(langs[1])]
        }
        # remove language token during generating
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super(SpeechToTextCifMutiContrast, self).build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )
    
    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        epoch = 1,
        task_cfg: FairseqDataclass = None,
        **kwargs
    ):
        pre_tokenizer = self.build_tokenizer()
        bpe_tokenizer = self.build_bpe()
        is_train_split = split.startswith("train")
        st_dataset = SpeechTextTripleDatasetCreator.from_tsv(
            self.cfg.data,
            self.data_cfg,
            split,
            self.src_dict,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed
        )
        text_dataset = None
        if self.cfg.external_mt_data is not None and is_train_split:
            text_dataset = self.load_langpair_dataset()
        if text_dataset is not None:
            muti_modal_dataset = [
                ModalityDatasetItem(
                    "speech_to_text",
                    st_dataset,
                    [self.cfg.max_audio_positions, self.cfg.max_target_positions],
                    self.cfg.max_audio_tokens,
                    self.cfg.batch_size
                ),
                ModalityDatasetItem(
                    "text_to_text",
                    text_dataset,
                    [self.cfg.max_source_positions, self.cfg.max_target_positions],
                    self.cfg.max_text_tokens,
                    self.cfg.batch_size
                )
            ]
            self.datasets[split] = MultiModalityDataset(muti_modal_dataset)
        else:
            self.datasets[split] = st_dataset
    
    def load_langpair_dataset(self):
        split = "train"
        src, tgt = self.cfg.lang_pairs.split("-")
        return load_langpair_dataset(
            self.cfg.external_mt_data,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=False,
            dataset_impl=None,
            upsample_primary=1,
            left_pad_source=False,
            left_pad_target=False,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=False,
            truncate_source=False,
            shuffle=True,
        )
    
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False
    ):
        # if not use external MT dataset, call upper class method
        if not isinstance(dataset, MultiModalityDataset):
            return super().get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache,
                skip_remainder_batch,
                grouped_shuffling,
                update_epoch_batch_itr
            )
    
    def build_dataset_for_inference(
        self,
        src_tokens: List[Tensor],
        src_lengths: List[int], **kwargs
    ):
        return SpeechTextTripleDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
    
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
        assert len(bleu.counts) == EVAL_BLEU_ORDER
        for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output
            
    def inference_step(
        self,
        generator,
        models,
        sample,
        prefix_tokens=None,
        constraints=None
    ):
        if self.cfg.lang_prefix_tok is None:
            prefix_tokens = None
        else: 
            prefix_tokens = self.tgt_dict.index(self.cfg.lang_prefix_tok)
            assert prefix_tokens != self.tgt_dict.unk_index
        with torch.no_grad():
            net_input = sample["net_input"]
            assert "src_tokens" in net_input, "Missing `src_tokens`"
            src_tokens = net_input["src_tokens"]
            B = src_tokens.size()[0]
            if prefix_tokens is not None:
                if isinstance(prefix_tokens, int):
                    prefix_tokens = torch.LongTensor([prefix_tokens]).unsqueeze(1)
                    prefix_tokens = prefix_tokens.expand(B, -1).to(src_tokens.device)
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints
            )
        
    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = self._inference_decode(gen_out[i][0]["tokens"])
            ref = self._inference_decode(
                utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                escape_unk=True
            )
            if self.cfg.lang_prefix_token is not None:
                hyp = hyp.replace(self.cfg.lang_prefix_tok, "")
                ref = ref.replace(self.cfg.lang_prefix_tok, "")
            hyps.append(hyp)
            refs.append(ref)
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def _inference_decode(self, tokens, escape_unk=False):
        str = self.tgt_dict.string(
            tokens.int().cpu(),
            self.cfg.eval_bleu_remove_bpe,
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP")
        )
        if self.bpe is not None:
            str = self.bpe.decode(str)
        if self.tokenizer:
            str = self.tokenizer.decode(str)
        return str
            
    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(path).shape[0] for path in lines]
        return lines, n_frames
    
    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def max_positions(self):
        ...
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:
            ...

        