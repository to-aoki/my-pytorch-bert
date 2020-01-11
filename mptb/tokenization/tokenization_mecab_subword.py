# Author Toshihiko Aoki
#
# This file is based on https://github.com/google-research/bert/blob/master/tokenization.py.
# Mecab tokenizer.
#
# Copyright 2018 The Google AI Language Team Authors.
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

import MeCab
from random import randint
from collections import OrderedDict
from tqdm import tqdm

CONTROL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']


class MeCabTokenizer(object):

    def __init__(
        self,
        dict_path='/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd',
        preprocessor=None,
        lemmatize=False
    ):
        self.tagger = MeCab.Tagger('-d ' + dict_path)
        self.tagger.parse('')
        self.preprocessor = preprocessor
        self.lemmatize = lemmatize

    def tokenize(self, text):
        text = self.preprocessor(text) if self.preprocessor is not None else text
        tokens = []
        for chunk in self.tagger.parse(text.rstrip()).splitlines()[:-1]:  # skip EOS
            if chunk == '' or '\t' not in chunk:  # often there is not include tab
                continue
            (surface, features) = chunk.split('\t')
            token = surface.strip()
            feature = features.split(',')
            if self.lemmatize and 6 < len(feature) and feature[6] != '*' and feature[6] != '':
                token = feature[6].strip()
            tokens.append(token)
        return tokens

    def pos(self, token):
        for chunk in self.tagger.parse(token).splitlines()[:-1]:  # skip EOS
            if chunk == '' or '\t' not in chunk:  # often there is not include tab
                continue
            _, features = chunk.split('\t')
            feature = features.split(',')
            pos = '[' + '_'.join(feature[:4]) + ']'  # e.g. [名詞_固有名詞_一般_*]
            return pos
        return '[UNK]'


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    with open(vocab_file, "r", encoding='utf-8') as reader:
        return token_vocab_build(reader)


def token_vocab_build(reader):
    vocab_dict = OrderedDict()
    index = 0
    for _, token in enumerate(tqdm(reader)):
        vocab_dict[token] = index
        index += 1
    return vocab_dict


def convert_by_vocab(vocab_dict, items, unk_info):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab_dict:
            output.append(vocab_dict[item])
        else:
            output.append(unk_info)
    return output


def convert_tokens_to_ids(vocab, tokens):
    """Id of <unk> is assumed as 0"""
    return convert_by_vocab(vocab, tokens, unk_info=1)


def convert_ids_to_tokens(inv_vocab, ids):
    """Token of unknown word is assumed as [UNK]"""
    return convert_by_vocab(inv_vocab, ids, unk_info='[UNK]')


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, preprocessor=None, max_chars_per_word=40, control_tokens=CONTROL_TOKENS):
        self.tokenizer = MeCabTokenizer(preprocessor=preprocessor)
        self.vocab = load_vocab(vocab_file)
        assert (0 < len(self.vocab))
        self.inv_vocab = {}
        self.control_len = 0
        self.max_chars_per_word = max_chars_per_word
        self.unk_id = 0
        for k, v in self.vocab.items():
            if k == '[PAD]':
                self.pad_idx = v
            if k == '[UNK]':
                self.unk_id = v
            if k in control_tokens:
                self.control_len += 1
            self.inv_vocab[v] = k

    def tokenize(self, text):
        output_tokens = []
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            chars = list(token)
            if len(chars) > self.max_chars_per_word:
                pos_token = self.tokenizer.pos(token)
                output_tokens.append(pos_token)
                continue

            not_found = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                correct_subword = None
                while start < end:
                    word = "".join(chars[start:end])
                    if start > 0:
                        subword = "##" + word
                    else:
                        subword = word
                    if subword in self.vocab:
                        correct_subword = subword
                        break
                    else:
                        pos_token = self.tokenizer.pos(word)
                        if pos_token in self.vocab:
                            correct_subword = pos_token
                            break
                    end -= 1

                if correct_subword is None:
                    not_found = True
                    break

                sub_tokens.append(correct_subword)
                start = end

            if not_found:
                output_tokens.append('[UNK]')
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens, unk_info=self.unk_id)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids, unk_info='[UNK]')

    def get_random_token_id(self):
        return randint(self.control_len + 1, len(self.inv_vocab) - 1)

    def __len__(self):
        return len(self.vocab)

