# Author Toshihiko Aoki
#
# This file is based on :
# https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py.
# https://github.com/pytorch/fairseq/blob/master/fairseq/data/dictionary.py.
# Roberta tokenizer.
#
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import re
from .gpt2_bpe_utils import get_encoder
import numpy as np


class RobertaDictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
            self,
            pad='[PAD]',
            eos='[SEP]',
            unk='[UNK]',
            bos='[CLS]',
            extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word, self.bos_word = unk, pad, eos, bos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:
        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f, ignore_utf_errors)
        return d

    def add_from_file(self, f, ignore_utf_errors=False):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        self.add_from_file(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx + 1:])
            self.indices[word] = len(self.symbols)
            self.symbols.append(word)
            self.count.append(count)

    def _load_meta(self, lines):
        return 0

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.
        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, 'bos_index'):
            sent = ' '.join(token_string(i) for i in tensor if (i != self.eos()) and (i != self.bos()))
        else:
            sent = ' '.join(token_string(i) for i in tensor if i != self.eos())
        return sent

    SPACE_NORMALIZER = re.compile(r"\s+")

    @classmethod
    def tokenize_line(cls, line):
        line = cls.SPACE_NORMALIZER.sub(" ", line)
        line = line.strip()
        return line.split()

    def encode_line(self, line, add_if_not_exist=True,
                    consumer=None, append_eos=False, reverse_order=False):
        tokens = self.tokenize_line(line)
        return self.encode_tokens(tokens, add_if_not_exist, consumer, append_eos, reverse_order)

    def encode_tokens(self, tokens, add_if_not_exist=True,
                    consumer=None, append_eos=False, reverse_order=False):
        if reverse_order:
            words = list(reversed(tokens))
        ids = []
        for i, word in enumerate(tokens):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids.append(idx)
        if append_eos:
            ids.append(self.eos_index)
        return ids


class FullTokenizer(object):

    """Runs end-to-end tokenziation."""

    def __init__(self, dict_path, encoder_json_path, vocab_bpe_path):
        self.roberta_dict = RobertaDictionary.load(dict_path)
        self.gpt2_get_encoder = get_encoder(encoder_json_path, vocab_bpe_path)

    def tokenize(self, text):
        return self.roberta_dict.tokenize_line(' '.join(map(str, self.gpt2_get_encoder.encode(text))))

    def convert_tokens_to_ids(self, tokens):
        return self.roberta_dict.encode_tokens(tokens)

    def convert_ids_to_tokens(self, ids):
        tokens = np.array(ids)
        if tokens[0] == self.roberta_dict.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.roberta_dict.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.gpt2_get_encoder.decode_tokens(map(int, self.roberta_dict.string(s).split())) for s in sentences]
        return sentences[0]

    def convert_text_to_ids(self, text):
        return self.roberta_dict.encode_line(
            self.roberta_dict.bos_word + ' ' +
            ' '.join(map(str, self.gpt2_get_encoder.encode(text))) + ' ' + self.roberta_dict.eos_word)

    def convert_ids_to_text(self, ids):
        tokens = np.array(ids)
        if tokens[0] == self.roberta_dict.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.roberta_dict.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.gpt2_get_encoder.decode(map(int, self.roberta_dict.string(s).split())) for s in sentences]
        return sentences

    def __len__(self):
        return len(self.roberta_dict)
