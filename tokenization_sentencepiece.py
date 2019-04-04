# coding=utf-8
#
# Author Toshihiko Aoki
#
# This file is based on
# https://raw.githubusercontent.com/yoheikikuta/bert-japanese/master/src/tokenization_sentencepiece.py.
# Add get_random_token and len and calculate control chars.
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

"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import sentencepiece as sp
import six
from random import randint
import unicodedata
from utils import replace_num_zero, replace_uri


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


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
    vocab = collections.OrderedDict()
    index = 0
    while True:
        token = convert_to_unicode(reader.readline())
        if not token:
            break
        token, _ = token.split("\t")
        token = token.strip()
        vocab[token] = index
        index += 1

    return vocab


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
    """Token of unknown word is assumed as <unk> according to sentencepiece"""
    return convert_by_vocab(inv_vocab, ids, unk_info="<unk>")


class FullTokenizer(object):

    """Runs end-to-end tokenziation."""

    def __init__(self, model_file, vocab_file, do_lower_case=True,
                 do_normalize=True,
                 form='NFKC',
                 do_num_zero=True,
                 do_convert_uri=True,
                 replace_uri_word='link'):
        self.tokenizer = SentencePieceTokenizer(model_file, do_lower_case=do_lower_case)
        self.vocab = load_vocab(vocab_file)
        assert(0 < len(self.vocab))
        self.inv_vocab = {}
        self.control_len = 1  # <unk>
        for k, v in self.vocab.items():
            if self.tokenizer.tokenizer.is_control(v):
                self.control_len += 1  # Control characters are focused at the top?
            self.inv_vocab[v] = k
        self.do_normalize = do_normalize
        self.form = form
        self.do_num_zero = do_num_zero
        self.do_convert_uri = do_convert_uri
        self.replace_uri_word = replace_uri_word

    def tokenize(self, text):
        if self.do_normalize:
            text = unicodedata.normalize(self.form, text)
        if self.do_num_zero:
            text = replace_num_zero(text)
        if self.do_convert_uri:
            text = replace_uri(text, self.replace_uri_word)
        split_tokens = self.tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Id of <unk> is assumed as 0 accroding to sentencepiece"""
        return convert_by_vocab(self.vocab, tokens, unk_info=3)  # <unk> = 0 -> [PAD]?

    def convert_ids_to_tokens(self, ids):
        """Token of unknown word is assumed as <unk> according to sentencepiece"""
        return convert_by_vocab(self.inv_vocab, ids, unk_info="[PAD]")  # <unk> = 0 -> [PAD]?
    
    # add for get random word
    def get_random_token_id(self):
        return randint(self.control_len+1, len(self.inv_vocab)-1)

    def __len__(self):
        return len(self.tokenizer)


class SentencePieceTokenizer(object):
    """Runs SentencePiece tokenization (from raw text to tokens list)"""

    def __init__(self, model_file=None, do_lower_case=True):
        if model_file is None:
            raise ValueError("You have to give a path of trained SentencePiece model.")        
        """Constructs a SentencePieceTokenizer."""
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.Load(model_file)
        print("Loaded a trained SentencePiece model.")
        self.do_lower_case = do_lower_case

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        if self.do_lower_case:
            text = text.lower()
        output_tokens = self.tokenizer.EncodeAsPieces(text)
        return output_tokens

