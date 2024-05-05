import json
import pickle
import re
from tqdm import tqdm

file = open("./data/prediction_test_all2all.json")
dict = json.load(file)
file.close()

file = open("./data/shared_task_test.jsonl")
lines = file.readlines()
file.close()
test_dict = {}
for line in lines:
    dict_ = json.loads(line)
    test_dict[int(dict_["id"])] = dict_["claim"]


with open('wiki.pickle', 'rb') as handle:
    wikipedia_dict = pickle.load(handle)

premises, hypothesis, ids = [], [], []
for key,value in tqdm(dict.items()):
    premises.append(test_dict[int(key)].strip())
    
    sents = []
    for lis in value[0:3]:
        print(lis[0], lis[1], len(wikipedia_dict[lis[0]]))
        text = wikipedia_dict[lis[0]][lis[1]]
        text = text.replace("\t", " ")
        text = re.sub(r'\s+', ' ', text)
        text = " ".join(text.split(" ")[1:]).strip()
        
        if text == '' or text == " ": continue
        sents.append(text)
    
    if len(sents) == 0: print("0 length")
    hypothesis.append(" ".join(sents))
    ids.append(int(key))

assert len(premises) == len(hypothesis)

file = open("./data/xnli/test-en.tsv", "w+")
for i in range(len(premises)):
    file.write(premises[i] + "\t" + hypothesis[i] + "\t" + "REFUTES\n")
file.close()

file = open("./data/xnli/test-en-withids.tsv", "w+")
for i in range(len(premises)):
    file.write(str(ids[i]) + "\t" + premises[i] + "\t" + hypothesis[i] + "\t" + "REFUTES\n")
file.close()