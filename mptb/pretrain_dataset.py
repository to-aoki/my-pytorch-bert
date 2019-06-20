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

import os
from random import random
from random import randint, shuffle, randrange

import torch
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import truncate_seq_pair, get_tokenizer
import copy
import pickle


class PretrainDataset(Dataset):

    def __init__(self, tokenizer, max_pos, dataset_path=None, documents=[], encoding="utf-8", on_memory=True):
        self.tokenizer = tokenizer
        self.max_pos = max_pos

        if dataset_path is None and len(documents) == 0:
            raise ValueError('dataset_path or documents require.')
        self.dataset_path = dataset_path

        self.on_memory = on_memory
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        # BERT reserved tokens
        self.pad_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        self.cls_id = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.mask_id = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

        self.corpus_lines = 0
        # load samples into memory
        if len(documents) > 0 or on_memory:
            self.all_documents = []
            doc = []
            if len(documents) > 0:
                for text in documents:
                    doc = self._load_text(doc, text)
            else:
                with open(dataset_path, "r", encoding=encoding) as reader:
                    for text in tqdm(reader, desc="Loading Dataset", total=self.corpus_lines):
                        doc = self._load_text(doc, text)
            # FIX to last rows ""... . last line is not "" and EOF
            if len(doc) > 0:
                self.all_documents.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_documents)

            if len(self.all_documents) is 0:
                raise ValueError(dataset_path + ' were not includes documents.')

        # load samples later lazily from disk
        else:
            with open(dataset_path, "r", encoding=encoding) as reader:
                for line in tqdm(reader, desc="Loading Dataset", total=self.corpus_lines):
                    if line.strip() == "":
                        self.num_docs += 1
                    else:
                        self.corpus_lines += 1

                # if doc does not end with empty line
                if line.strip() != "":
                    self.num_docs += 1

            self.file = open(dataset_path, "r", encoding=encoding)
            self.random_file = open(dataset_path, "r", encoding=encoding)

    def _load_text(self, doc, text):
        text = text.strip()
        if text == "":
            if len(doc) > 0:  # FIX to last rows ""...
                self.all_documents.append(doc)
                doc = []
                # remove last added sample because there won't be a subsequent line anymore in the doc
                self.sample_to_doc.pop()
        else:
            # store as one sample
            sample = {"doc_id": len(self.all_documents), "line": len(doc)}
            self.sample_to_doc.append(sample)
            tokens = self.tokenizer.tokenize(text) if self.tokenizer is not None else text
            doc.append(self.tokenizer.convert_tokens_to_ids(tokens))
            self.corpus_lines += 1
        return doc

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
                self.file = open(self.dataset_path, "r", encoding=self.encoding)

        t1, t2, is_next_label = self.random_sent(item)

        # transform sample to features
        features = self.convert_example_to_features(t1, t2, is_next_label, self.max_pos)
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
                while t1 == "":
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
            t1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t1))
            t2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t2))

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
            self.random_file = open(self.dataset_path, "r", encoding=self.encoding)
            line = self.random_file.__next__().strip()
        line = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
        return line
    
    def get_random_token_id(self):
        return self.tokenizer.get_random_token_id()

    def convert_example_to_features(
            self, tokens_a, tokens_b, is_next_label, max_pos, short_seq_prob=0.1, masked_lm_prob=0.15):
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

        target_max_pos = max_pos - 3 if tokens_b else max_pos - 2

        tokens_a_ids = copy.copy(tokens_a)
        tokens_b_ids = copy.copy(tokens_b)
        # However, sequences to minimize the mismatch between pre-training and fine-tuning.
        if random() < short_seq_prob:
            target_max_pos = randint(2, target_max_pos)
        truncate_seq_pair(tokens_a_ids, tokens_b_ids, target_max_pos)

        # Add Special Tokens
        tokens_a_ids.insert(0, self.cls_id)
        tokens_b_ids.append(self.sep_id)
        if len(tokens_b_ids) != 0:
            tokens_b_ids.append(self.sep_id)
        else:
            tokens_b_ids = []

        tokens = copy.copy(tokens_a_ids)
        tokens.extend(copy.copy(tokens_b_ids))
        # Add next sentence segment
        segment_ids = [0] * len(tokens_a_ids) + [1] * len(tokens_b_ids)

        # mask prediction calc
        mask_prediction = int(round(len(tokens) * masked_lm_prob))

        mask_candidate_pos = [i for i, token in enumerate(tokens) if token != self.cls_id and token != self.sep_id]
        # masked and random token
        shuffle(mask_candidate_pos)
        for pos in mask_candidate_pos[:mask_prediction]:
            if random() < 0.8:    # 80%
                tokens[pos] = self.mask_id
            elif random() < 0.5:  # 10%
                tokens[pos] = self.get_random_token_id()
            # 10% not mask and not modify

        input_ids = tokens
        input_mask = [1]*len(input_ids)
        label_ids = tokens_a_ids+tokens_b_ids

        # zero padding
        num_zero_pad = max_pos - len(input_ids)
        input_ids.extend([self.pad_id]*num_zero_pad)
        segment_ids.extend([0]*num_zero_pad)
        input_mask.extend([0]*num_zero_pad)
        label_ids.extend([self.pad_id]*num_zero_pad)

        return [input_ids,  segment_ids, input_mask, is_next_label, label_ids]

    def __str__(self):
        name, _ = os.path.splitext(os.path.basename(self.dataset_path))
        return name


class PretrainDataGeneration(object):

    def __init__(
        self,
        dataset_path=None,
        output_path=None,
        vocab_path='tests/sample_text.vocab',
        sp_model_path='tests/sample_text.model',
        max_pos=512,
        epochs=1,
        tokenizer_name='google',
    ):
        tokenizer = get_tokenizer(
            vocab_path=vocab_path, sp_model_path=sp_model_path, name=tokenizer_name)

        if max_pos < 5:
            import statistics
            with open(dataset_path, 'r', newline="\n", encoding="utf-8") as data:
                tokens = list(map(self.tokenizer.tokenize, data.readlines()))
                median_pos = round(statistics.median(list(map(lambda x: len(x), tokens))))
            max_pos = median_pos * 2 + 3  # [CLS]a[SEP]b[SEP]
            print("max_pos (median):", max_pos)

        self.dataset = PretrainDataset(
            tokenizer=tokenizer, max_pos=max_pos, dataset_path=dataset_path, on_memory=True
        )
        self.output_path = output_path
        self.epochs = epochs

    def generate(self):
        sampler = RandomSampler(self.dataset)
        gen_dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=1)
        for e in range(self.epochs):
            iter_bar = tqdm(
                gen_dataloader, "generate pretrain input file")
            with open(self.output_path + '_' + str(e) + '.pickle', 'ab+') as f:
                for step, batch in enumerate(iter_bar):
                    pickle.dump(batch, f)


class PretensorPretrainDataset(Dataset):

    def __init__(self, dataset_path=None, length=2642016):
        self.file = open(dataset_path, "rb")
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        object = pickle.load(self.file)
        return object[0][0], object[1][0], object[2][0], object[3][0], object[4][0]
