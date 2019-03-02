# This file is based on https://github.com/yoheikikuta/bert-japanese/blob/master/src/train-sentencepiece.py.
# Fix to read the json-file to make the setting method same as other code.

import json
import glob
from typing import NamedTuple
import sentencepiece as sp


class Config(NamedTuple):
    text_dir: str = "/work/data/wiki/"
    prefix: str = "/work/model/wiki-ja"
    vocab_size: int = 32000
    ctl_symbols: str = "[PAD],[CLS],[SEP],[MASK]"

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r", encoding="UTF-8")))


def _get_text_file(text_dir):
    file_list = glob.glob(f'{text_dir}/*.txt')
    files = ",".join(file_list)
    return files


def train(text_dir, prefix, vocab_size, ctl_symbols):
    files = _get_text_file(text_dir)
    command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} --control_symbols={ctl_symbols}'
    sp.SentencePieceTrainer.Train(command)


def main(sp_cfg="config/test_sp.json"):
    cfg = Config.from_json(sp_cfg)
    print(cfg)
    train(cfg.text_dir, cfg.prefix, cfg.vocab_size, cfg.ctl_symbols)


if __name__ == '__main__':
    main()