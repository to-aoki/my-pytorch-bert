# Author Toshihiko Aoki
#
# This file is based on https://github.com/google-research/bert/blob/master/tokenization.py.
# Ginza (spacy) tokenizer.
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

import spacy
from random import randint
from mojimoji import han_to_zen
from collections import OrderedDict
from tqdm import tqdm

CONTROL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
# https://github.com/megagonlabs/ginza/blob/master/ja_ginza/sudachi_tokenizer.py
UPOS_TOKENS = ['SYM', 'INTJ', 'SPACE', 'ADJ', 'ADP', 'PART', 'SCONJ', 'AUX', 'NOUN',
               'PRON', 'VERB', 'ADV', 'PUNCT', 'PROPN', 'NUM', 'DET']


def create_vocab(create_file_path, control_tokens=CONTROL_TOKENS, upos_tokens=UPOS_TOKENS):
    with open(create_file_path, "w", encoding='utf-8', newline='\n') as f:
        nlp = spacy.load('ja_ginza_nopn')
        for _, vector in enumerate(tqdm(list(nlp.vocab.vectors))):
            f.write(nlp.vocab.strings[vector] + '\n')
        for control in control_tokens:
            f.write(control+'\n')
        for upos in upos_tokens:
            f.write('[' + upos + ']' + '\n')
    return len(nlp.vocab.vectors) + len(control_tokens) + len(upos_tokens)


class GinzaTokenizer(object):

    def __init__(
        self,
        preprocessor=None,
        lemmatize=False,
        collect_futures=[],
        upos_tokens=UPOS_TOKENS
    ):
        self.nlp = spacy.load('ja_ginza_nopn')
        self.collect_futures = collect_futures
        self.preprocessor = preprocessor
        self.lemmatize = lemmatize
        self.upos_tokens = upos_tokens

    def tokenize(self, text):
        text = self.preprocessor(text) if self.preprocessor is not None else text
        text = han_to_zen(text)  # dict use zenkaku
        tokens = []
        doc = self.nlp(text.rstrip())
        for token in doc:
            if token.has_vector:
                word = token.orth_.strip()
                if self.lemmatize:
                    word = token.lemma_
            else:
                if token.pos_ in self.upos_tokens:
                    word = '[' + token.pos_ + ']'
                else:
                    word = '[UNK]'
            tokens.append(word)
        return tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    try:
        import tensorflow as tf
        with tf.gfile.GFile(vocab_file, "r") as reader:
            return token_vocab_build(reader)

    except ImportError:
        with open(vocab_file, "r", encoding='utf-8') as reader:
            return token_vocab_build(reader)


def token_vocab_build(reader):
    vocab_dict = OrderedDict()
    index = 0
    for _, token in enumerate(tqdm(reader)):
        word = word.strip()
        vocab_dict[word] = index
        index += 1
    return vocab_dict


def convert_by_vocab(vocab_dict, items, unk_info, nlp=None):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab_dict:
            output.append(vocab_dict[item])
        else:
            output.append(unk_info)
    return output


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, preprocessor=None, control_tokens=CONTROL_TOKENS):
        self.tokenizer = GinzaTokenizer(preprocessor=preprocessor)
        self.vocab = load_vocab(vocab_file)
        assert (0 < len(self.vocab))
        self.inv_vocab = {}
        self.control_start = 0
        for k, v in self.vocab.items():
            if v == control_tokens[0]:
                self.control_start = k
            self.inv_vocab[v] = k

    def tokenize(self, text):
        split_tokens = self.tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens, unk_info=100002)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids, unk_info='[UNK]')

    # add for get random word
    def get_random_token_id(self):
        return randint(0, self.control_start)

    def __len__(self):
        return len(self.vocab)



