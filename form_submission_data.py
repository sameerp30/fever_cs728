import json

file = open("./data/claim2id.json")
claim2id_dict = json.load(file)
file.close()

file = open("./data/xnli/test-en-results.tsv")
lines = file.readlines()
file.close()

label_dict = {0: "REFUTES", 1: "SUPPORTS", 2: "NOT ENOUGH INFO"}

predictions = [label_dict[int(line.split("\t")[1])] for line in lines]

file = open("./data/xnli/test-en.tsv")
lines = file.readlines()
file.close()

claims = [line.split("\t")[0].strip() for line in lines]

file = open("./data/xnli/test-en-withids.tsv")
lines = file.readlines()
file.close()

ids = [line.split("\t")[0].strip() for line in lines]

assert len(predictions) == len(claims) == len(ids)

# file = open("query2pred_lines_test_5_10.json")
#file = open("all2all_retrieval_results.json")
file = open("./data/prediction_test_all2all.json")
claim2evidence_dict = json.load(file)
file.close()

# file = open("./data/shared_task_test.jsonl")
# lines = file.readlines()
# file.close()
# test_claim_id_dict = {}
# for line in lines:
#     dict_ = json.loads(line)
#     test_claim_id_dict[dict_["claim"]] = dict_["id"]

# print(len(test_claim_id_dict.keys()))


#evidences = [claim2evidence_dict[str(test_claim_id_dict[claim])][0:3] for claim in claims]
evidences = [claim2evidence_dict[str(id)][0:3] for id in ids]

assert len(predictions) == len(claims) == len(evidences)


final_objects = []
# for i in range(len(claims)):
#     claim_ids = claim2id_dict[claims[i]]
#     for claim_id in claim_ids:
#         dict = {"id": claim_id, "predicted_label": predictions[i], "predicted_evidence": evidences[i]}
#         final_objects.append(dict)
for i in range(len(ids)):
    id = ids[i]
    dict = {"id": id, "predicted_label": predictions[i], "predicted_evidence": evidences[i]}
    final_objects.append(dict)
    # for claim_id in claim_ids:
        

# ids_present = [dict["id"] for dict in final_objects]

with open("prediction.jsonl", 'w') as f:
    for item in final_objects:
        f.write(json.dumps(item) + "\n")


# file = open("./data/shared_task_test.jsonl")
# test_lines = file.readlines()
# file.close()

# for line in test_lines:
#     dict = json.loads(line)
#     id = dict["id"]
#     if id not in ids_present: print("id found", id)