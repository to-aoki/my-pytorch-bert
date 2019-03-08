# coding=utf-8
#
# Author Toshihiko Aoki
# This file is based on
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_lm_finetuning.py.
# This uses the part of BERTDataset.
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""for BERT pre-training dataset."""

from random import random
from random import randint, shuffle, randrange

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import truncate_seq_pair


class PretrainDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, max_pos,
                 encoding="utf-8", corpus_lines=None,  on_memory=True):
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.max_pos = max_pos
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_documents = []
            doc = []
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    line = line.strip()
                    if line == "":
                        if len(doc) > 0:  # FIX to last rows ""...
                            self.all_documents.append(doc)
                            doc = []
                            # remove last added sample because there won't be a subsequent line anymore in the doc
                            self.sample_to_doc.pop()
                    else:
                        # store as one sample
                        sample = {"doc_id": len(self.all_documents),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append(line)
                        self.corpus_lines += 1

            # FIX to last rows ""... . last line is not "" and EOF
            if len(doc) > 0:
                self.all_documents.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_documents)

            if len(self.all_documents) is 0:
                raise ValueError(corpus_path + ' were not includes documents.')

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        if line.strip() is "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() is not "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1  # self.num_docs = num_spaces

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        t1, t2, is_next_label = self.random_sent(item)

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        tokens_b = self.tokenizer.tokenize(t2)

        # transform sample to features
        features = self.convert_example_to_features(tokens_a, tokens_b, is_next_label, self.max_pos)

        return [torch.tensor(x, dtype=torch.long) for x in features]

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        if random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1

        return t1, t2, label

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""

        assert isinstance(item, int), 'item only support int(index) access.'
        assert item < self.corpus_lines, 'item index out range corpus.'

        if self.on_memory:
            sample = self.sample_to_doc[item]
            t1 = self.all_documents[sample["doc_id"]][sample["line"]]
            t2 = self.all_documents[sample["doc_id"]][sample["line"]+1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1, t2
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "" :
                    t1 = self.file.__next__().strip()
                    t2 = self.file.__next__().strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                t2 = self.file.__next__().strip()
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t2 == "" or t1 == "":
                    t1 = self.file.__next__().strip()
                    t2 = self.file.__next__().strip()
                    self.current_doc = self.current_doc+1
            self.line_buffer = t2

        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = randrange(len(self.all_documents))
                rand_doc = self.all_documents[rand_doc_idx]
                line = rand_doc[randrange(len(rand_doc))]
            else:
                rand_index = randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                # pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            # check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = self.random_file.__next__().strip()
            # keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = self.random_file.__next__().strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = self.random_file.__next__().strip()
        return line
    
    def get_random_word(self):
        return self.tokenizer.get_random_token()

    def convert_example_to_features(self, tokens_a, tokens_b, is_next_label, max_pos, short_seq_prob=0.1, masked_lm_prob=0.15):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param tokens_a: str, example tokens.
        :param tokens_b: str, example next tokens.
        :param is_next_label: int, is next label.
        :param max_pos: int, maximum length of sequence.
        :param short_seq_prob: float, Probability of creating sequences which are shorter than the maximum length.
        :param masked_lm_prob: float, Masked LM probability.
        :return: features
        """

        target_max_pos = max_pos - 3

        # However, sequences to minimize the mismatch between pre-training and fine-tuning.
        if random() < short_seq_prob:
            target_max_pos = randint(2, target_max_pos)
        truncate_seq_pair(tokens_a, tokens_b, target_max_pos)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        
        # Add next sentence segment
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1) 

        # mask prediction calc
        mask_prediction = int(round(len(tokens) * masked_lm_prob))

        mask_candidate_pos = [i for i, token in enumerate(tokens) if token != '[CLS]' and token != '[SEP]']
        # masked
        shuffle(mask_candidate_pos)
        for pos in mask_candidate_pos[:mask_prediction]:
            if random() < 0.8:    # 80%
                tokens[pos] = '[MASK]'
            elif random() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()
            # 10% not mask and not modify

        # tokens indexing
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)
        label_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]'])

        # zero padding
        num_zero_pad = max_pos - len(input_ids)
        input_ids.extend([0]*num_zero_pad)
        segment_ids.extend([0]*num_zero_pad)
        input_mask.extend([0]*num_zero_pad)
        label_ids.extend([0]*num_zero_pad)

        return [input_ids,  segment_ids, input_mask, is_next_label, label_ids]
