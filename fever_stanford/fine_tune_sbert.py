import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import random
random.seed(10)
from sentence_transformers import InputExample
from random import sample
import re
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import losses
import json
from sentence_transformers import evaluation
import torch


fever_wiki = load_dataset("fever", 'wiki_pages')

with open('/raid/nlp/sameer/fever/data/fever-data/train.jsonl') as f:
    data = [json.loads(line) for line in f]


wiki_id2lines = {}
for dump in tqdm(fever_wiki['wikipedia_pages']):
    if len(dump['text']) != 0:
        wiki_id2lines[dump['id']] = [dump['text'], dump['lines']]


##### preparing train samples ############################################################
        
claim2evi_sent = {}
claim2evi_id = {}
for i in range(0,len(data)):
    if data[i]['label'] == 'NOT ENOUGH INFO':
        continue
    gold = []
    gold_line = []
    claim2evi_sent[data[i]['claim']] = {}
    claim2evi_id[data[i]['claim']] = {}
    for j in range(len(data[i]['evidence'])):
        for k in range(len(data[i]['evidence'][j])):
            
            if data[i]['evidence'][j][k][2] is not None:
                claim2evi_sent[data[i]['claim']][data[i]['evidence'][j][k][2]] = set()
                claim2evi_id[data[i]['claim']][data[i]['evidence'][j][k][2]] = set()
            if data[i]['evidence'][j][k][3] is not None:
                try:
                    text = wiki_id2lines[data[i]['evidence'][j][k][2]][1].split("\n")[data[i]['evidence'][j][k][3]]
                    text = text.replace("\t", " ").strip(",. ")
                    text = re.sub(r'\s+', ' ', text)
                    text =  " ".join(text.split()[1:])
                    claim2evi_sent[data[i]['claim']][data[i]['evidence'][j][k][2]].add(text)
                    claim2evi_id[data[i]['claim']][data[i]['evidence'][j][k][2]].add(data[i]['evidence'][j][k][3])
                except:
                    continue

train_examples = []

for claim in claim2evi_id:
    for doc_id in claim2evi_id[claim]:
        if doc_id not in wiki_id2lines:
            continue
        lines_original = wiki_id2lines[doc_id][1].split("\n")
        lines = []
        for line in lines_original:
            text = line.replace("\t"," ").strip(",. ")
            text = re.sub(r'\s+', ' ', text)
            text = " ".join(text.split()[1:])
            lines.append(text)
        for pos_id in claim2evi_id[claim][doc_id]:
            pos_line = lines[pos_id]
            lines = [lines[j] for j in range(len(lines)) if len(lines[j])>0 and lines[j].isnumeric()==False and j not in claim2evi_id[claim][doc_id]]
            neg = sample(lines,min(len(lines), 8))
            for m in range(0,len(neg)):
                train_examples.append(InputExample(texts=[claim, pos_line, neg[m]]))


########### preparing dev samples ###########################
                
'''with open('/raid/nlp/sameer/fever/data/fever-data/dev.jsonl') as f:
    data = [json.loads(line) for line in f]

claim2evi_sent = {}
claim2evi_id = {}
for i in range(0,len(data)):
    if data[i]['label'] == 'NOT ENOUGH INFO':
        continue
    gold = []
    gold_line = []
    claim2evi_sent[data[i]['claim']] = {}
    claim2evi_id[data[i]['claim']] = {}
    for j in range(len(data[i]['evidence'])):
        for k in range(len(data[i]['evidence'][j])):
            
            if data[i]['evidence'][j][k][2] is not None:
                claim2evi_sent[data[i]['claim']][data[i]['evidence'][j][k][2]] = set()
                claim2evi_id[data[i]['claim']][data[i]['evidence'][j][k][2]] = set()
            if data[i]['evidence'][j][k][3] is not None:
                try:
                    text = wiki_id2lines[data[i]['evidence'][j][k][2]][1].split("\n")[data[i]['evidence'][j][k][3]]
                    text = text.replace("\t", " ").strip(",. ")
                    text = re.sub(r'\s+', ' ', text)
                    text =  " ".join(text.split()[1:])
                    claim2evi_sent[data[i]['claim']][data[i]['evidence'][j][k][2]].add(text)
                    claim2evi_id[data[i]['claim']][data[i]['evidence'][j][k][2]].add(data[i]['evidence'][j][k][3])
                except:
                    continue


queries = []
positives = []
negatives = []

for claim in claim2evi_id:
    for doc_id in claim2evi_id[claim]:
        if doc_id not in wiki_id2lines:
            continue
        lines_original = wiki_id2lines[doc_id][1].split("\n")
        lines = []
        for line in lines_original:
            text = line.replace("\t"," ").strip(",. ")
            text = re.sub(r'\s+', ' ', text)
            text = " ".join(text.split()[1:])
            lines.append(text)
        for pos_id in claim2evi_id[claim][doc_id]:
            pos_line = lines[pos_id]
            lines = [lines[j] for j in range(len(lines)) if len(lines[j])>0 and lines[j].isnumeric()==False and j not in claim2evi_id[claim][doc_id]]
            neg = sample(lines,min(len(lines), 8))
            for m in range(0,len(neg)):
                negatives.append(neg[m])
                positives.append(pos_line)
                queries.append(claim)'''

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

word_embedding_model = models.Transformer('distilroberta-base')

## Step 2: use a pool function over the token embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

## Join steps 1 and 2 using the modules argument
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model = model.to(torch.device("cuda:5"))
print("model loaded---------------------")

train_loss = losses.TripletLoss(model=model)
# evaluator = evaluation.EmbeddingSimilarityEvaluator(queries, positives, negatives)

# ... Your other code to load training data

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,output_path="/raid/nlp/sameer/fever/sbert_model/distilroberta_10_neg",
)

# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)