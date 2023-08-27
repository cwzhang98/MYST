import os
import logging
import json
from pathlib import Path
from fairseq.data import Dictionary, encoders
from argparse import Namespace
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.data.audio.data_cfg import S2TDataConfig

logger = logging.getLogger(__name__)


@register_task("wav2vec_asr_fine_tuning")
class Wav2VecAsrFineTuning(LegacyFairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument("--data", help="manifest root path")
        parser.add_argument("--config-yaml", type=str, default="config.yaml",
                            help="Configuration YAML filename (absolute path)")
        parser.add_argument("--max-audio-tokens", default=1000000, type=int, metavar="N",
                            help="max batch of tokens in audio sequences")
        parser.add_argument("--max-audio-positions", default=6000, type=int, metavar="N",
                            help="max number of tokens in the source audio sequence")
        # use different form of target sequence
        parser.add_argument("--target-type", type, choices=['phoneme', 'subword'], default='phoneme',
                            help='token type of target sequence')
        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        # space
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        # set None when using phoneme sequence
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        # set False
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        # decode settings during infernece
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        # print dev samples
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # set None or 'sentencepiece'
        parser.add_argument('--eval-bleu-bpe', type=str, metavar='BPE',
                            default=None,
                            help='args for building the bpe, if needed')

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        self.sequence_generator = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        dict_path = Path(args.data, args.langpairs) / data_cfg.vocab_filename
        assert os.path.isfile(dict_path), f"Dict not found: {dict_path}"
        tgt_dict = Dictionary.load(dict_path)
        logger.info(f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}")
        # training subset filename should be start with 'train'
        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
            
        if data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return cls(args, tgt_dict)

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) == "space", (
                "--eval-bleu-detok is required if using --eval-bleu; "
            )
            # if subword sequence
            if args.target_type == "subword":
                # detokenizer_args and eval_bleu_detok should be {} and 'space'
                detokenizer_args = json.loads(getattr(args, "eval_bleu_detok_args", None) or "{}")
                self.tokenizer = encoders.build_tokenizer(
                    Namespace(tokenizer=getattr(args, "eval_bleu_detok", None), **detokenizer_args)
                )
                if args.eval_bleu_bpe is not None:
                    self.bpe = self.build_bpe(args)
                else:
                    self.bpe = None         
        # beam search args
        gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
        # decode generator
        self.sequence_generator = self.build_generator([model], Namespace(**gen_args))

        return model

    def build_tokenizer(self, args):
            logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def max_positions(self):
        return self.args.max_audio_positions, self.args.max_source_positions

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        pass

    def valid_step(self, sample, model, criterion):
        """validation with scareBleu evaluation"""
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        pass

    def reduce_metrics(self, logging_outputs, criterion):
        pass




