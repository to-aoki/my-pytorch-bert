# Author Toshihiko Aoki
#
# This file is based on https://github.com/google-research/bert/blob/master/tokenization.py.
# Juman(pyknp) tokenizer.
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

from pyknp import Juman
from random import randint
from collections import OrderedDict
from tqdm import tqdm
from .preprocessing import *
from .utils import separate_japanese_doc
from .tokenization import WordpieceTokenizer, BasicTokenizer

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

# vocab build (use Juman++ v2.0.0-rc2 and subword-nmt)
# http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB


class JumanTokenizer(object):

    def __init__(
        self,
        preprocessor=None,
        stopwords=[],
    ):
        self.jumanpp = Juman()
        self.preprocessor = preprocessor
        self.stopwords = stopwords

    def tokenize(self, text, input_max_bytes=4096):
        text = self.preprocessor(text) if self.preprocessor is not None else text
        sentences = separate_japanese_doc(text.strip())  # avoid limit input
        tokens = []
        for sentence in sentences:
            if sentence == '':
                continue
            loop_sentence = sentence
            is_tokenize_continue = True
            while is_tokenize_continue:
                sub_text = loop_sentence
                while len(sub_text.encode()) > input_max_bytes:
                    sub_text = sub_text[:-1]
                for mrph in self.jumanpp.analysis(sub_text):
                    token = mrph.midasi.strip()
                    if token == '':
                        continue
                    if token in self.stopwords:
                        continue
                    tokens.append(token)
                is_tokenize_continue = False
                if len(loop_sentence) != len(sub_text):
                    is_tokenize_continue = True
                    loop_sentence = loop_sentence[len(sub_text)-1:]
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
    for index, word in enumerate(tqdm(reader)):
        word = word.rstrip('\n')
        vocab_dict[word] = index
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
        self.tokenizer = JumanTokenizer(preprocessor=preprocessor)
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {}
        self.control_len = 0
        for k, v in self.vocab.items():
            if v in control_tokens:
                self.control_len += 1
            self.inv_vocab[v] = k

        self.basic_tokenizer = BasicTokenizer()
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for juman_token in self.tokenizer.tokenize(text):
            for sub_token in self.basic_tokenizer.tokenize(juman_token, is_juman=True):
                for sub_sub_token in self.wordpiece_tokenizer.tokenize(sub_token):
                    split_tokens.append(sub_sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens, unk_info=1)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids, unk_info='[UNK]')

    # add for get random word
    def get_random_token_id(self):
        return randint(self.control_len + 1, len(self.inv_vocab) - 1)

    def __len__(self):
        return len(self.vocab)
