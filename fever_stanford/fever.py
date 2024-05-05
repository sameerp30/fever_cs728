# Adapted and modified from https://github.com/sheffieldnlp/fever-baselines/tree/master/src/scripts
# which is adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.

# Adapted and modified from nli.py, sst.py in https://github.com/cgpotts/cs224u/ 
# by Prof. Christopher Potts for CS224u, Stanford, Spring 2018




import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import random
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import utils
from doc_db import DocDB

DB_PATH = 'data/single/fever.db'
MAT_PATH = 'data/index/tfidf-count-ngram=1-hash=16777216.npz'

class Oracle(object):
    def __init__(self, 
            db_path=DB_PATH,
            mat_path=MAT_PATH):
        self.db_path = db_path
        self.mat_path = mat_path
        self.db = DocDB(db_path = self.db_path)
        self.mat, metadata = utils.load_sparse_csr(self.mat_path)

        # doc_freqs, hash_size, ngram, doc_dict
        for k, v in metadata.items():
            setattr(self, k, v)


    def __str__(self):
        return """FEVER Oracle\nDatabase path = {}\nTerm-Document matrix path = {}""".format(
            self.db_path, self.mat_path)

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items()}
        return """"FEVER Oracle({})""".format(d)


    def closest_docs(self, query, k=3):
        temp = self.mat.transpose().dot(utils.text2mat(query, self.hash_size, vector=True).transpose())
        inds = pd.Series(temp.toarray().squeeze()).nlargest(k).index
        return [self.doc_dict[1][ind] for ind in inds]


    def doc_ids2texts(self, doc_ids):
        return [self.db.get_doc_text(doc_id) for doc_id in doc_ids]

    def get_sentence(self, doc_id, sent_num):
        temp = sent_tokenize(self.db.get_doc_text(doc_id))
        if len(temp) > sent_num:
            return temp[sent_num]
        else:
            return temp[-1]

    def choose_sents_from_doc_ids(self, query, doc_ids, k=3):
        id_tuple = []
        texts = []
        for doc_id in doc_ids:
            sents = sent_tokenize(self.db.get_doc_text(doc_id))
            for j, sent in enumerate(sents):
                id_tuple.append((doc_id,j))
                texts.append(sent)
        chosen_sents = utils.closest_sentences(query, texts, self.hash_size, k=k)
        return {id_tuple[i]:sent for i, sent in chosen_sents.items()}
            
    def read(self, query, num_sents=3, num_docs=3):
        doc_ids = self.closest_docs(query, k=num_docs)
        return self.choose_sents_from_doc_ids(query, doc_ids, k=num_sents)


class Example(object):
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __str__(self):
        return """{}\n{}\n{}""".format(
            self.claim, self.verifiable, self.label)

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"FEVER Example({})""".format(d)

    def get_evidence_ids_for_retrieval_test(self):
        if not hasattr(self, 'label') or self.label == 'NOT ENOUGH INFO':
            return None
        return [(ev[2], ev[3]) for evi in self.evidence for ev in evi]

    def get_evidence_ids(self):
        return [(ev[2], ev[3]) for evi in self.evidence for ev in evi]


class Reader(object):
    def __init__(self,
            src_filename,
            samp_percentage=None,
            random_state=None):
        self.src_filename = src_filename
        self.samp_percentage = samp_percentage
        self.random_state = random_state

    def read(self):
        random.seed(self.random_state)
        for line in open(self.src_filename):
            if (not self.samp_percentage) or random.random() <= self.samp_percentage:
                d = json.loads(line)
                ex = Example(d)
                yield ex

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"FEVER Reader({})""".format(d)


FEVER_HOME = os.path.join("data", "fever-data")

class TrainReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "train.jsonl")
        super().__init__(src_filename, **kwargs)


class DevReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "dev.jsonl")
        super().__init__(src_filename, **kwargs)

class TestReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "test.jsonl")
        super().__init__(src_filename, **kwargs)

class SampledTrainReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "train_sampled.jsonl")
        super().__init__(src_filename, **kwargs)


class SampledDevReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "dev_sampled.jsonl")
        super().__init__(src_filename, **kwargs)

class SampledTestReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "test_sampled.jsonl")
        super().__init__(src_filename, **kwargs)




def build_dataset_from_scratch(reader, phi, oracle, vectorizer=None, vectorize=True):
    feats = []
    labels = []
    raw_examples = []
    
    total_len = len(set(reader.read()))
    for ex in tqdm(reader.read(), total=total_len, unit="examples", desc = 'Reading from dataset'):
        claim = ex.claim
        evidence = oracle.read(claim).values()
        label = ex.label
        d = phi(claim, evidence)
        feats.append(d)
        labels.append(label)
        raw_examples.append((claim, evidence))
    if vectorize:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=True)
            feat_matrix = vectorizer.fit_transform(feats)
        else:
            feat_matrix = vectorizer.transform(feats)
    else:
        # feat_matrix = feats
        feat_matrix = sp.vstack(feats)
    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}



def build_dataset(reader, phi, oracle, vectorizer=None, vectorize=True):
    feats = []
    labels = []
    raw_examples = []
    
    total_len = len(set(reader.read()))
    for ex in tqdm(reader.read(), total=total_len, unit="examples", desc = 'Reading from dataset'):
        claim = ex.claim
        ev_ids = ex.get_evidence_ids()
        sents = [oracle.get_sentence(ev_id[0], ev_id[1]) for ev_id in ev_ids]
        d = phi(claim, sents)
        feats.append(d)
        labels.append(ex.label)
        raw_examples.append((claim, sents))
    if vectorize:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=True)
            feat_matrix = vectorizer.fit_transform(feats)
        else:
            feat_matrix = vectorizer.transform(feats)
    else:
        # feat_matrix = feats
        feat_matrix = sp.vstack(feats)
    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}