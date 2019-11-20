# coding=utf-8
#
# Author Toshihiko Aoki
#
# This file is based on
# https://raw.githubusercontent.com/yoheikikuta/bert-japanese/master/src/tokenization_sentencepiece.py.
# Added unknown token support using ginza/spacy POS.
#
# Original Author yoheikikuta
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

"""Tokenization Sentencepiece and ginza POS """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sentencepiece as sp
import struct

import spacy
from random import randint


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    try:
        import tensorflow as tf
        with tf.io.gfile.GFile(vocab_file, "r") as reader:
            return token_vocab_build(reader)
            
    except ImportError:
        with open(vocab_file, "r", encoding='utf-8') as reader:
            return token_vocab_build(reader)


def token_vocab_build(reader):
    vocab = collections.OrderedDict()
    vocab_score = collections.OrderedDict()
    index = 0
    while True:
        token = reader.readline()
        if not token:
            break
        token, score = token.split("\t")
        token = token.strip()
        score = float(score)
        vocab[token] = index
        vocab_score[index] = score
        index += 1

    return vocab, vocab_score


def convert_by_vocab(vocab, items, unk_info):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(unk_info)
    return output


def convert_tokens_to_ids(vocab, tokens):
    """Id of <unk> is assumed as 0 accroding to sentencepiece"""
    return convert_by_vocab(vocab, tokens, unk_info=0)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids, unk_info='<unk>')


class FullTokenizer(object):

    """Runs end-to-end tokenziation."""

    def __init__(self, model_file, vocab_file, preprocessor=None):
        self.tokenizer = SentencePieceTokenizer(model_file, preprocessor=preprocessor)
        self.nlp = spacy.load('ja_ginza')
        self.vocab, self.vocab_score = load_vocab(vocab_file)
        assert(0 < len(self.vocab))
        self.inv_vocab = {}
        self.control_len = 1  # 0 : <unk>
        for k, v in self.vocab.items():
            if self.tokenizer.tokenizer.is_control(v):
                self.control_len += 1  # Control characters are focused at the top?
            self.inv_vocab[v] = k

    def tokenize(self, text):
        split_tokens = self.tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        output = []
        for item in tokens:
            append_id = self.vocab.get(item, None)
            if append_id is not None:
                output.append(append_id)
            else:
                found = False
                try:
                    doc = self.nlp(item)
                    for sent in doc.sents:
                        for token in sent:
                            if token.pos_ is not None and token.orth_ == item:
                                pos = '[' + token.pos_ + ']'
                                append_id = self.vocab.get(pos, None)
                                if append_id is not None:
                                    output.append(append_id)
                                    found = True
                                    break
                        if found:
                            break
                except struct.error:
                    # '𥝱'
                    import warnings
                    warnings.warn(item)
                    pass
                if not found:
                    output.append(self.vocab['<unk>'])
        return output

    def convert_ids_to_tokens(self, ids):
        """Token of unknown word is assumed as <unk> according to sentencepiece"""
        return convert_by_vocab(self.inv_vocab, ids, unk_info="<unk>")

    # add for get random word
    def get_random_token_id(self):
        return randint(self.control_len+1, len(self.inv_vocab)-1)

    def __len__(self):
        return len(self.tokenizer)


class SentencePieceTokenizer(object):
    """Runs SentencePiece tokenization (from raw text to tokens list)"""

    def __init__(self, model_file=None, preprocessor=None):
        if model_file is None:
            raise ValueError("You have to give a path of trained SentencePiece model.")        
        """Constructs a SentencePieceTokenizer."""
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.Load(model_file)
        self.preprocessor = preprocessor
        print("Loaded a trained SentencePiece model.")

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self.preprocessor(text) if self.preprocessor else text
        output_tokens = self.tokenizer.EncodeAsPieces(text)
        return output_tokens

