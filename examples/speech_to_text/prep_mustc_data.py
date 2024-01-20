#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from data_utils import (
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    load_df_from_tsv,
    save_df_to_tsv,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "speaker", "tgt_text", "src_text_pho", "src_text_raw", "src_lang", "tgt_lang"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            if _lang == "en":
                pho_txt_path = txt_root / f"{split}_pho.{_lang}"  # train split src text: load phoneme text
                with open(pho_txt_path) as f:
                    pho_utterances = [r.strip() for r in f]
            raw_txt_path = txt_root / f"{split}.{_lang}"  # other: load raw text
            with open(raw_txt_path) as f:
                raw_utterances = [r.strip() for r in f]      
            assert len(segments) == len(raw_utterances)
            if _lang == "en":
                for i, (pu, ru) in enumerate(zip(pho_utterances, raw_utterances)):
                    segments[i][f'{_lang}_pho'] = pu
                    segments[i][f'{_lang}_raw'] = ru
            else:
                for i, u in enumerate(raw_utterances):
                    segments[i][_lang] = u
                
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            # segments in single wav file are sorted by relative offset
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                # get segments info
                offset = int(float(segment["offset"]) * sample_rate)  # number_of_frames == sample_rate * time
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"  # e.g. wav_1
                self.data.append(
                    (
                        wav_path.as_posix(),  # abs path
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en_pho"],
                        segment["en_raw"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[str, int, int, int, str, str, str, str, str]:
        return self.data[n]

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    """
    1. no need to process wavfile since using raw audio as input
    2. generate manifest tsv file for every split, and filter it
    3. build tokenizer and process text, and save tokenizer model
    4. generate dictionary
    5. generate yaml config file
    """
    root = Path(args.data_root).absolute()
    lang = args.lang
    cur_root = root / f"en-{lang}"
    assert cur_root.is_dir(), f"{cur_root} is not exist."

    # Generate TSV manifest
    train_text_mono = []
    train_text_bi = []
    # process all splits
    for split in MUSTC.SPLITS:
        print(f"Fetching en-{lang} split {split}...")
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = MUSTC(args.data_root, lang, split)
        for wav_path, offset, n_frames, sample_rate, src_utt_pho, src_utt_raw, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(f"{wav_path}:{offset}:{n_frames}")
            manifest["n_frames"].append(n_frames)
            manifest["src_text_pho"].append(src_utt_pho)
            manifest["src_text_raw"].append(src_utt_raw)
            manifest["tgt_text"].append(tgt_utt)
            manifest["speaker"].append(speaker_id)
            manifest["src_lang"].append("en")
            manifest["tgt_lang"].append(lang)
        if is_train_split:
            train_text_mono.extend(manifest["tgt_text"])  # train spm only using target text
            train_text_bi.extend(manifest["tgt_text"])
            train_text_bi.extend(manifest["src_text_raw"])
        df = pd.DataFrame.from_dict(manifest)
        # filer manifest
        df = filter_manifest_df(df, is_train_split=is_train_split, max_n_frames=480000, min_n_frames=1000)
        save_df_to_tsv(df, cur_root / f"{split}_st.tsv")

    # add extra mt training data to train sp model
    # if args.extra_mt_data_path:
    #     print("add extra mt data to train text")
    #     mt_data_path = Path(args.extra_mt_data_path) / "en-de"
    #     assert mt_data_path.is_dir()
    #     with open(mt_data_path / "train.de") as f:
    #         train_text.extend([r.strip() for r in f])

    # Generate vocab
    vocab_size_str_mono = "" if args.vocab_type == "char" else str(args.vocab_size_mono)
    vocab_size_str_bi = "" if args.vocab_type == "char" else str(args.vocab_size_bi)
    spm_filename_prefix_mono = f"spm_{args.vocab_type}_{vocab_size_str_mono}_st"
    spm_filename_prefix_bi = f"spm_{args.vocab_type}_{vocab_size_str_bi}_st"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text_mono:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix_mono,
            args.vocab_type,
            args.vocab_size_mono,
            special_symbols=[f"<lang:{lang}>"],
            accept_language=[f"{lang}"]
        )
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text_bi:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix_bi,
            args.vocab_type,
            args.vocab_size_bi,
            special_symbols=[f"<lang:{lang}>", "<lang:en>"],
            accept_language=[f"{lang}", "en"]
        )
    
    # Generate config YAML
    gen_config_yaml(
        cur_root,
        args.lang,
        spm_filename_prefix_bi + ".model",
        yaml_filename=f"config_st.yaml",
        prepend_tgt_lang_tag=True,
        prepend_src_lang_tag=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--lang", type=str, default="de",
                        choices=["de", "es", "fr", "it", "nl", "pt", "ro", "ru"])

    parser.add_argument("--src-tgt-joint-dict", action="store_true", help="generate joint dictionary")  # False
    parser.add_argument("--vocab-type", default="unigram", required=True, type=str,
                        choices=["bpe", "unigram", "char"])
    parser.add_argument("--extra-mt-data-path", type=str, default="")
    parser.add_argument("--vocab-size-mono", default=5000, type=int)
    parser.add_argument("--vocab-size-bi", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
