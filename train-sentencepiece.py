# This file is based on https://github.com/yoheikikuta/bert-japanese/blob/master/src/train-sentencepiece.py.
# Fix to read the json-file to make the setting method same as other code.
#
# Author : Yohei Kikuta
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sentencepiece training."""

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
        with open(file, "r", encoding="UTF-8") as reader:
            config = json.load(reader)
        return cls(**config)


def _get_text_file(text_dir):
    file_list = glob.glob(f'{text_dir}/*.txt')
    files = ",".join(file_list)
    return files


def train(text_dir, prefix, vocab_size, ctl_symbols):
    files = _get_text_file(text_dir)
    command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} --control_symbols={ctl_symbols}'
    sp.SentencePieceTrainer.Train(command)


def main(config_poth="config/test_sp.json"):
    config = Config.from_json(config_poth)
    print(config)
    train(config.text_dir, config.prefix, config.vocab_size, config.ctl_symbols)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SentencePiece training.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines training.', nargs='?',
                        type=str, default='tests/test_sp.json')
    args = parser.parse_args()
    main(args.config_path)

