from datasets import load_dataset
from tqdm import tqdm
import re
import pickle
import json

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

def get_hypo(dict):
    evidence_dict = {}
    
    for evidence in dict["evidence"]:
        evidence = evidence[0]
        wiki_url = evidence[2]
        sent_id = evidence[3]
        
        if wiki_url in evidence_dict: evidence_dict[wiki_url].append(sent_id)
        else: evidence_dict[wiki_url] = [sent_id]
    
    sents = []
    for key,value in evidence_dict.items():
        value = list(set(value))
        value.sort()
        
        if key in wikipedia_dict:
            wiki_sents = wikipedia_dict[key]
            for idx in value:
                text = wiki_sents[idx]
                text = text.replace("\t", " ")
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
            train_nei_claims.append(claim)
            continue
        
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
            dev_nei_claims.append(claim)
            continue
        
        hypo = get_hypo(dict)
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