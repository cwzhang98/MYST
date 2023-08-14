from pathlib import Path
from typing import *
from argparse import ArgumentParser
from examples.speech_to_text.data_utils import S2TDataConfigWriter

def gen_w2v_pt_config_yaml(
        manifest_root: Path,
        yaml_filename: str = "config.yaml",
        prepend_tgt_lang_tag: bool = True
):
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()
    gen_w2v_pt_config_yaml(
        Path(args.root),
        args.yaml_filename,
        False
    )
