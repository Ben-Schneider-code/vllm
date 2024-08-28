import os

PATH_TO_QUERY_JSON = os.path.expanduser("~/M-BEIR/query/union_train/mbeir_union_up_train.jsonl")
PATH_TO_CAND_JSON =  os.path.expanduser("~/M-BEIR/cand_pool/global/mbeir_union_train_cand_pool.jsonl")
PATH_TO_DICT_JSON = os.path.expanduser("~/viz_dict.pickle")

DATASET_CAN_NUM_UPPER_BOUND = 10000000  # Maximum number of candidates per dataset
DATASET_QUERY_NUM_UPPER_BOUND = 500000  # Maximum number of queries per dataset

OUTPUT_DIR = os.path.expanduser("~/viz_output")

def hash_did(did):
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id

def hash_qid(qid):
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id

import orjson as json
import torch

with open(PATH_TO_QUERY_JSON) as f:
    query_dict = {}
    for line in f:
        obj = json.loads(line)
        query_dict[hash_qid(obj["qid"])] = obj

with open(PATH_TO_CAND_JSON) as f:
    cand_dict = {}
    for line in f:
        obj = json.loads(line)
        cand_dict[hash_did(obj["did"])] = obj

sample_data = torch.load(PATH_TO_DICT_JSON)

print("hello")

counter = 0

for batch in sample_data:
    for j in batch["meta"]:
        print("HELLO")