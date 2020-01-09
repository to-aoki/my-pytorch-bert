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
UPOS_TOKENS = ['NOUN', 'PROPN', 'SYM', 'NUM', 'SPACE']
NER_TOKENS = ['LOC', 'DATE', 'ORG', 'PERSON', 'PRODUCT']


def create_vocab(create_file_path,
                 control_tokens=CONTROL_TOKENS, upos_tokens=UPOS_TOKENS, ner_tokens=NER_TOKENS, vocab_size=32000,
                 lang='ja_ginza'):
    with open(create_file_path, "w", encoding='utf-8', newline='\n') as f:
        nlp = spacy.load(lang)
        num_tokens = vocab_size - (len(control_tokens) + len(upos_tokens) + len(ner_tokens))
        skiped = []
        for _, vector in enumerate(tqdm(list(nlp.vocab.vectors))):
            word = nlp.vocab.strings[vector]
            doc = nlp(word)
            found = False
            ner = ''
            pos = ''
            for ent in doc.ents:
                if ent.label_ is not None:
                    ner = ent.label_
                    if ner in ner_tokens:
                        found = True
                        break
            for sent in doc.sents:
                for token in sent:
                    if token.pos_ is not None:
                        pos = token.pos_
                        if pos in upos_tokens:
                            found = True
                            break
            if found:
                skiped.append({'word': word, 'vector': vector, 'ner': ner, 'pos': pos})
            else:
                f.write(word + '\t' + str(vector) + '\t' + ner + '\t' + pos + '\n')
                num_tokens -= 1
                if num_tokens == 0:
                    break

        if num_tokens != 0:
            for skip in skiped:
                if skip['pos'] == 'SPACE':
                    continue
                f.write(skip['word'] + '\t' + str(skip['vector']) + '\t' + skip['ner'] + '\t' + skip['pos'] + '\n')
                num_tokens -= 1
                if num_tokens == 0:
                    break

        for ner in ner_tokens:
            f.write('[' + ner + ']' + '\n')
        for upos in upos_tokens:
            f.write('[' + upos + ']' + '\n')
        for control in control_tokens:
            f.write(control+'\n')

    return vocab_size


class GinzaTokenizer(object):

    def __init__(
        self,
        preprocessor=None,
        lemmatize=True,
        collect_futures=[],
        lang='ja_ginza'
    ):
        self.nlp = spacy.load(lang)
        self.collect_futures = collect_futures
        self.preprocessor = preprocessor
        self.lemmatize = lemmatize

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
                word = '[UNK]'
            tokens.append(word)
        return tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    with open(vocab_file, "r", encoding='utf-8') as reader:
        return token_vocab_build(reader)


def token_vocab_build(reader):
    vocab_dict = OrderedDict()
    index = 0
    for _, token in enumerate(tqdm(reader)):
        vocabs = token.split("\t")
        word = vocabs[0].rstrip()
        vocab_dict[word] = index
        index += 1
    return vocab_dict


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, preprocessor=None,
                 control_tokens=CONTROL_TOKENS,
                 upos_tokens=UPOS_TOKENS, ner_tokens=NER_TOKENS):
        self.tokenizer = GinzaTokenizer(preprocessor=preprocessor)
        self.vocab = load_vocab(vocab_file)
        assert (0 < len(self.vocab))
        self.inv_vocab = {}
        self.control_start = 0
        for k, v in self.vocab.items():
            if k == control_tokens[0]:
                self.control_start = v
            if k == '[UNK]':
                self.unk_id = v
            self.inv_vocab[v] = k
        self.upos_tokens = upos_tokens
        self.ner_tokens = ner_tokens

    def tokenize(self, text):
        split_tokens = self.tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        output = []
        for item in tokens:
            token_id = self.vocab.get(item)
            if token_id is not None:
                output.append(token_id)
            else:
                doc = self.tokenizer.nlp(item)
                found = False
                for ent in doc.ents:
                    if ent.label_ is not None:
                        ner = ent.label_
                        ner_id = self.vocab.get('[' + ner + ']')
                        if ner_id is not None:
                            output.append(ner_id)
                            found = True
                            break
                if found:
                    continue
                for sent in doc.sents:
                    for token in sent:
                        if token.pos_ is not None:
                            pos = token.pos_
                            pos_id = self.vocab.get('[' + pos + ']')
                            if pos_id is not None:
                                output.append(pos_id)
                                found = True
                                break
                    if found:
                        break
                if found:
                    continue
                output.append(self.unk_id)
        return output

    def convert_ids_to_tokens(self, ids):
        output = []
        for item in ids:
            token = self.inv_vocab[item]
            if token is not None:
                output.append(token)
            else:
                output.append('[UNK]')
        return output

    # add for get random word
    def get_random_token_id(self):
        return randint(0, self.control_start)

    def __len__(self):
        return len(self.vocab)



