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

import os
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


def _get_text_file(text_dir, suffix=''):
    file_list = glob.glob(f'{text_dir}/*{suffix}')
    files = ",".join(file_list)
    return files


def train(text_dir, prefix, vocab_size, ctl_symbols, is_bpe=False, suffix='', subword_vocab_path='test_subword.vocab'):
    files = _get_text_file(text_dir, suffix)
    if is_bpe:
        command = f'--input={files} --model_type=bpe --model_prefix={prefix} --vocab_size={vocab_size} ' \
                  f'--control_symbols={ctl_symbols}  ' \
                  f'--character_coverage=1.0 --normalization_rule_name=identity ' \
                  f'--pad_id=0 --unk_id=1 --bos_id=-1 eos_id=-1'
    else:
        command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} --control_symbols={ctl_symbols} '\
                  f'--add_dummy_prefix=False --treat_whitespace_as_suffix=True --bos_id=-1 eos_id=-1'
    sp.SentencePieceTrainer.Train(command)
    if is_bpe:
        build_wordpiece_vocab(prefix + '.vocab', subword_vocab_path, ctl_symbols)


def build_wordpiece_vocab(sp_vocab_file, build_vocab_path, ctl_symbols='[PAD],[CLS],[SEP],[MASK]'):
    with open(sp_vocab_file) as sp_vocab, \
            open(build_vocab_path, 'w') as wordpiece_vocab:
        reserved = ctl_symbols.split(',')
        for line in sp_vocab:
            sp_token, _ = line.split('\t')
            if sp_token == '<unk>':
                output_token = '[UNK]'
            elif sp_token == '<pad>':
                output_token = '[PAD]'
            elif sp_token in reserved:
                output_token = sp_token
            elif sp_token.startswith('\u2581'):
                output_token = sp_token[1:]
            else:
                output_token = '##' + sp_token
            wordpiece_vocab.write(output_token + '\n')


def main(config_poth="config/test_sp.json", is_bpe=False, suffix='', subword_vocab_path='test_subword.vocab'):
    config = Config.from_json(config_poth)
    print(config)
    train(config.text_dir, config.prefix, config.vocab_size, config.ctl_symbols, is_bpe, suffix, subword_vocab_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SentencePiece training.', usage='%(prog)s [options]')
    parser.add_argument('--config_path', help='JSON file path for defines training.', nargs='?',
                        type=str, default='tests/test_sp.json')
    parser.add_argument('--bpe', action='store_true', help='BPE training for subword.')
    parser.add_argument('--subword_vocab_path', type=str, default='tests/test_subword.vocab')
    parser.add_argument('--suffix',  type=str, default='.txt')
    args = parser.parse_args()
    main(args.config_path, args.bpe, args.suffix, args.subword_vocab_path)

