from datasets import load_dataset
from tqdm import tqdm
import re
import pickle
import json
import random

fever_wiki = load_dataset("fever", 'wiki_pages')
fever_nli = load_dataset("fever", 'v1.0')

print(fever_nli)
print(fever_wiki)

# uncomment below lines to generate the dictionary
"""wikipedia_dict = {}
i = 0
for dict in tqdm(fever_wiki["wikipedia_pages"]):
    if len(dict["text"]) != 0:
        wikipedia_dict[dict["id"]] = dict["lines"].split("\n")

print(len(wikipedia_dict))

with open('wiki.pickle', 'wb') as handle:
    pickle.dump(wikipedia_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

with open('wiki.pickle', 'rb') as handle:
    wikipedia_dict = pickle.load(handle)

file = open("./tfidf_pred_train_not_enough.json")
train_nei = json.load(file)
file.close()

file = open("./tfidf_pred_dev_not_enough.json")
dev_nei = json.load(file)
file.close()

random.seed(42)

def get_hypo(dict):
    visited_dict = {}
    
    sents = []
    
    for evidence_list in dict["evidence"]:
        for evidence in evidence_list:
            wiki_url = evidence[2]
            sent_id = evidence[3]
            
            if wiki_url + "_" + str(sent_id) in visited_dict: continue
            visited_dict[wiki_url + "_" + str(sent_id)] = 1
            
            if wiki_url in wikipedia_dict:
                wiki_sents = wikipedia_dict[wiki_url]
                text = wiki_sents[sent_id]
                text = text.replace("\t", " ")
                text = re.sub(r'\s+', ' ', text)
                text = " ".join(text.split(" ")[1:]).strip()
                
                if text == '' or text == " ": continue
                
                sents.append(text)
                
                
    
    
    if len(sents) == 0: return None
    else: return " ".join(sents)

def form_nli_data():
    file = open("./train.jsonl")
    train_lines = file.readlines()
    file.close()

    file = open("./paper_dev.jsonl")
    dev_lines = file.readlines()
    file.close()
    
    train_nei_claims, dev_nei_claims = [], []
    train_premises, train_hypos, train_labels = [], [], []
    dev_premises, dev_hypos, dev_labels = [], [], []

    for line in tqdm(train_lines):
        dict = json.loads(line)
        claim = dict["claim"]
        label = dict["label"]
        
        if label == "NOT ENOUGH INFO":
            if claim in train_nei:
                url = train_nei[claim][0]
                if url in wikipedia_dict:
                    wiki_sents = wikipedia_dict[url]
                    if len(wiki_sents) == 0: continue
                    # randomly pick between 2-6 sentences
                    num = random.randint(min(2, len(wiki_sents)), min(7, len(wiki_sents)))
                    random.shuffle(wiki_sents)
                    sents = []
                    for i in range(num):
                        text = wiki_sents[i]
                        text = text.replace("\t", " ")
                        text = re.sub(r'\s+', ' ', text)
                        text = " ".join(text.split(" ")[1:]).strip()
                        
                        if text == '' or text == " ": continue
                        
                        sents.append(text)
                    hypo = " ".join(sents)
                    
                
            # train_nei_claims.append(claim)
            # continue
        else:
            hypo = get_hypo(dict)
        
        if hypo is None: continue
        
        train_premises.append(claim)
        train_hypos.append(hypo)
        train_labels.append(label)

    for line in tqdm(dev_lines):
        dict = json.loads(line)
        claim = dict["claim"]
        label = dict["label"]
        
        if label == "NOT ENOUGH INFO":
            # dev_nei_claims.append(claim)
            # continue
            if claim in dev_nei:
                url = dev_nei[claim][0]
                if url in wikipedia_dict:
                    wiki_sents = wikipedia_dict[url]
                    if len(wiki_sents) == 0: continue
                    # randomly pick between 2-6 sentences
                    num = random.randint(min(2, len(wiki_sents)), min(7, len(wiki_sents)))
                    random.shuffle(wiki_sents)
                    sents = []
                    for i in range(num):
                        text = wiki_sents[i]
                        text = text.replace("\t", " ")
                        text = re.sub(r'\s+', ' ', text)
                        text = " ".join(text.split(" ")[1:]).strip()
                        
                        if text == '' or text == " ": continue
                        
                        sents.append(text)
                    hypo = " ".join(sents)
        else: hypo = get_hypo(dict)
        
        if hypo is None: continue
        
        dev_premises.append(claim)
        dev_hypos.append(hypo)
        dev_labels.append(label)

    file = open("./data/train.tsv", "w+")
    for i in range(len(train_premises)):
        file.write(train_premises[i] + "\t" + train_hypos[i] + "\t" + train_labels[i] + "\n")
    file.close()

    file = open("./data/dev.tsv", "w+")
    for i in range(len(dev_premises)):
        file.write(dev_premises[i] + "\t" + dev_hypos[i] + "\t" + dev_labels[i] + "\n")
    file.close()


if __name__ == "__main__":
    form_nli_data()