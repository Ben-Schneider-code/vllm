# ------------
# Evals against flickr30k
# ------------

import os
import sys
from tqdm import tqdm
import json
from evaluate.embed_function import get_model_with_embed_function

supported_models = ["abcQwenVL-Instruct"]
import torch

def intersect(l1, l2):
    return len(set.intersection(set(l1), set(l2))) > 0

def get_topk_candidates(queries, candidates, k=3):

    query_ids, query_embs = zip(*queries)      
    candidate_ids, candidate_embs = zip(*candidates)
    query_stack = torch.cat(query_embs, dim=0).cuda()
    candidate_stack = torch.cat(candidate_embs, dim=0).cuda()
    scores = (query_stack @ candidate_stack.T).cpu()
    topk_values, topk_indices = torch.topk(scores, k=k, dim=1)

    # Build the results dictionary
    results = {}
    for i, q_id in enumerate(query_ids):
        # topk_indices[i] gives the indices of the top-k candidates for this query
        top_candidate_ids = [candidate_ids[idx] for idx in topk_indices[i]]
        results[q_id] = top_candidate_ids

    return results

def load(model_type, pretrain_model_path, instruct_model_path):
    return get_model_with_embed_function(model_type, pretrain_model_path, instruct_model_path=instruct_model_path)

def eval_vg_instruct(fxn):
    
    eval_path = os.environ.get("VG_EVAL", None)
    assert(eval_path is not None)
    with open(os.path.join(eval_path, "eval_dataset.json"), "rb") as f:
        ds_json = json.loads(f.read())
    
    ds_json = [item for sublist in ds_json for item in sublist]

    #assert len([i["phrase"] for i in ds_json]) == len(set([i["phrase"] for i in ds_json]))
    
    images = [( f"{i["image"]}.jpg", fxn( os.path.join(eval_path,"images",f"{i["image"]}.jpg"), dtype="image", instruction=i["instruction"])) for i in tqdm(ds_json)]
    text = [(str(i["id"]),fxn(i["phrase"], dtype="text")) for i in tqdm(ds_json)]


    # i2t
    for topk in [1,5,10]:
        cand = get_topk_candidates(images, text, topk)
        acc = 0
        for x, y in zip(images, text):
            targets = [y[0]]
            preds = cand[x[0]]
            if intersect(preds, targets): acc += 1
        acc = acc / len(images)
        print(f"i2t top{topk} is {acc:.4f}")

    # # t2i
    # for topk in [1,5,10]:
    #     cand = get_topk_candidates(text, images, topk)
    #     acc = 0
    #     cntr = 0
    #     # each image has multiple associated text descriptions
    #     for x in test:
    #         for t in x["sentences"]:
    #             targets = [x["image"]]
    #             preds = cand[t["raw"]]
    #             if intersect(preds, targets): acc += 1
    #             cntr += 1
    #     acc = acc / cntr
    #     print(f"t2i top{topk} is {acc:.4f}")

def main(model_type: str, pretrain_model_path: str, instruct_model_path: str):
    fxn = load(model_type, pretrain_model_path, instruct_model_path)
    eval_vg_instruct(fxn)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
