import os 
import json
from tqdm import tqdm
import pickle

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)
    return data


PATH = './wiki-pages/'

merged = {}

for file in tqdm(os.listdir(PATH)):
    file_path = PATH+file
    jsonl_data = read_jsonl(file_path)
    
    for i in tqdm(range(len(jsonl_data))):
        if jsonl_data[i]['id'] in merged:
            print('Skip', jsonl_data[i]['id'])
        else:
            merged[jsonl_data[i]['id']] = {}
            text = jsonl_data[i]['text']
            lines = jsonl_data[i]['lines'].splitlines()
            
            merged[jsonl_data[i]['id']]['text'] = text 
            merged[jsonl_data[i]['id']]['lines'] = lines

wiki_dict = {}

for key,value in merged.items():
    wiki_dict[key] = merged[key]['lines']


# with open('/raid/nlp/pranavg/meet/CS728/fever_cs728/data/processed_wiki-pages.json', 'w') as json_file:
#     json.dump(merged, json_file)
with open('wiki_fever.pickle', 'wb') as handle:
    pickle.dump(wiki_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
