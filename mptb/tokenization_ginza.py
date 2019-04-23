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
from collections import OrderedDict
from tqdm import tqdm
from .preprocessing import *

CONTROL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python 3")

# TODO spacy train(vocab create)


class GinzaTokenizer(object):

    def __init__(
        self,
        preprocessor=None,
        lemmatize=True,
        stopwords=[],
        collect_futures=[]
    ):
        self.ginza = spacy.load('ja_ginza_nopn')
        self.collect_futures = collect_futures
        self.preprocessor = preprocessor
        self.lemmatize = lemmatize
        self.stopwords = stopwords

    def tokenize(self, text):
        text = self.preprocessor(text) if self.preprocessor is not None else text
        tokens = []
        doc = self.ginza(text.rstrip())
        for sent in doc.sents:
            for token in sent:
                word = token.orth_.strip()
                if word == '':
                    continue
                if word in self.stopwords:
                    continue
                if self.lemmatize:
                    word = token.lemma_
                if not self.collect_futures:
                    tokens.append(word)
                else:
                    if token.pos_ in self.collect_futures:
                        tokens.append(word)
                # handle dep
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
        word, _ = token.split("\t")
        word = word.strip()
        vocab_dict[word] = index
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

    def __init__(self, vocab_file, preprocessor=None, control_tokens=CONTROL_TOKENS):
        self.tokenizer = GinzaTokenizer(preprocessor=preprocessor)
        self.vocab = load_vocab(vocab_file)
        assert (0 < len(self.vocab))
        self.inv_vocab = {}
        self.control_len = 0
        for k, v in self.vocab.items():
            if v in control_tokens:
                self.control_len += 1
            self.inv_vocab[v] = k

    def tokenize(self, text):
        split_tokens = self.tokenizer.tokenize(convert_to_unicode(text))
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens, unk_info=0)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids, unk_info='[UNK]')

    # add for get random word
    def get_random_token_id(self):
        return randint(self.control_len + 1, len(self.inv_vocab) - 1)

    def __len__(self):
        return len(self.vocab)



