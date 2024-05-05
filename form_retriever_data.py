import random

random.seed(42)

file = open("./data/retriever/train-en-entire.tsv")
lines = file.readlines()
file.close()

print("Total train examples are ", len(lines))

query_dict = {}

for line in lines:
    query = line.split("\t")[0]
    evidence = line.split("\t")[1]
    label = line.split("\t")[2].replace("\n", "")
    
    if query in query_dict:
        if label == "positive": query_dict[query]["positive"].append(evidence)
        elif label == "negative": query_dict[query]["negative"].append(evidence)
    else:
        query_dict[query] = {"positive": [], "negative": []}
        if label == "positive": query_dict[query]["positive"].append(evidence)
        elif label == "negative": query_dict[query]["negative"].append(evidence)

queries = list(query_dict.keys())
print(len(queries))

random.shuffle(queries)

queries = queries[0:20000]

file = open("./data/retriever/train-en.tsv", "w+")
for query in queries:
    positives = query_dict[query]["positive"]
    negatives = query_dict[query]["negative"]
    
    for pos in positives:
        file.write(query + "\t" + pos + "\t" + "positive\n")
    
    for neg in negatives:
        file.write(query + "\t" + neg + "\t" + "negative\n")
file.close()