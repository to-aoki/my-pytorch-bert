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
from collections import Counter, OrderedDict
from tqdm import tqdm
from math import log
import sys
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


def create_vocab(text_file_path, create_file_path, min_freq=0, limit_vocab_length=-1,
                 use_tfidf=False, collect_futures=[], control_tokens=CONTROL_TOKENS):
    with open(text_file_path, "r", encoding='utf-8') as reader:
        return text_to_vocab(reader, create_file_path, min_freq=min_freq, limit_vocab_length=limit_vocab_length,
                             use_tfidf=use_tfidf, collect_futures=collect_futures, control_tokens=control_tokens)


def text_to_vocab(
        reader, create_file_path, min_freq=0, max_freq=-1, limit_vocab_length=-1,
        use_tfidf=False, collect_futures=[], control_tokens=CONTROL_TOKENS):

    mecab_tokenizer = MeCabTokenizer(collect_futures=collect_futures)

    min_freq = max(min_freq, 1) if not use_tfidf else min_freq
    max_freq = min(max_freq, sys.maxsize) if not use_tfidf else max_freq
    counter = Counter()
    tfd = list()
    dft = Counter()

    for _, line in enumerate(tqdm(reader, 'term counts')):
        sentence = convert_to_unicode(line)
        if not sentence:
            break
        tf = Counter()
        tokens = mecab_tokenizer.tokenize(sentence)
        for word in tokens:
            counter[word] += 1
            tf[word] += 1
        for unique in list(set(tokens)):
            dft[unique] += 1
        tfd.append(tf)

    sort_freq = sorted(counter.items(), key=lambda tup: tup[0])
    sort_freq.sort(key=lambda tup: tup[1], reverse=True)
    if use_tfidf:
        size = len(tfd)
        tfidf_max = dict.fromkeys(dft.keys(), 0.)
        for word, df in tqdm(dft.items(), 'tf-idf calculate'):
            for tf in tfd:
                count = tf.get(word)
                if count is None:
                    continue
                tfidf = count/len(tf) * (log(size/df) + 1)
                if tfidf_max[word] < tfidf:
                    tfidf_max[word] = tfidf

        sort_freq = sorted(tfidf_max.items(), key=lambda tup: tup[0])
        sort_freq.sort(key=lambda tup: tup[1], reverse=True)

    with open(create_file_path, "w", encoding='utf-8', newline='\n') as f:
        for control in control_tokens:
            del counter[control]
            f.write(control+'\t0\n')
        size = len(control_tokens)

        for word, freq in sort_freq:
            if freq < min_freq or (max_freq != -1 and max_freq < freq):
                continue
            f.write(word+'\t'+str(freq)+'\n')
            size += 1
            if limit_vocab_length != -1 and limit_vocab_length < size:
                break
    return size


class MeCabTokenizer(object):

    def __init__(
        self, args='-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd',
        preprocessor=None,
        lemmatize=False,
        stopwords=[],
        collect_futures=[]
    ):
        self.tagger = MeCab.Tagger(args)
        self.tagger.parse('')
        self.collect_futures = collect_futures
        self.preprocessor = preprocessor
        self.lemmatize = lemmatize
        self.stopwords = stopwords

    def tokenize(self, text):
        text = self.preprocessor(text) if self.preprocessor is not None else text
        tokens = []
        for chunk in self.tagger.parse(text.rstrip()).splitlines()[:-1]:  # skip EOS
            if chunk == '' or '\t' not in chunk:  # often there is not include tab
                continue
            (surface, features) = chunk.split('\t')
            token = surface.strip()
            feature = features.split(',')
            pos_text = '.'.join(feature[:4])
            if self.lemmatize and 6 < len(feature) and feature[6] != '*' and feature[6] != '':
                token = feature[6].strip()
            if token in self.stopwords:
                continue

            if len(self.collect_futures) == 0:
                tokens.append(token)
            else:
                if feature[0] in self.collect_futures:
                    tokens.append(token)
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
        self.tokenizer = MeCabTokenizer(preprocessor=preprocessor)
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

