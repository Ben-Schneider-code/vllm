# ------------
# Evals against the mmeb classification set
# Minus N24News (as it is not i2t)
# ------------

import os
import sys
from tqdm import tqdm
from evaluate.embed_function import get_model_with_embed_function
from datasets import load_dataset
supported_models = ["abcQwenVL"]
splits = ["ImageNet-1K","HatefulMemes","VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet"]
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

def load(model_type: str, model_path: str):
    if model_type == "abcQwenVL":
        return get_model_with_embed_function(model_type, model_path)

def unroll_split(ds):
    labels = ds[0]["tgt_text"]
    labels_set = set(labels)
    query = []

    for item in ds:
        target_list = item["tgt_text"]
        query.append({"img": item["qry_img_path"], "target": target_list[0]})

        # assert that we can reuse the same embeddings
        assert set(target_list) == labels_set
    return query, labels

def eval_mmeb_classification(fxn, split_name):

    mmeb_path = os.environ["MMEB_EVAL"]
    ds = load_dataset("TIGER-Lab/MMEB-eval", split_name)["test"]
    q, c = unroll_split(ds)

    images = [(i["img"],fxn(os.path.join(mmeb_path,i["img"]), dtype="image")) for i in tqdm(q, disable=True)]
    text = [(i,fxn(f"A photo of {i}.", dtype="text")) for i in tqdm(c, disable=True)]

    # i2t
    print(f"{split_name}")
    for topk in [1]:
        cand = get_topk_candidates(images, text, topk)
        acc = 0
        for query in q:
            targets = [query["target"]]
            preds = cand[query["img"]]
            if intersect(preds, targets): acc += 1
        acc = acc / len(images)
        print(f"i2t top{topk} is {acc:.4f}")


def main(model_type: str, model_path: str):
    fxn = load(model_type, model_path)
    for benchmark in splits:
        eval_mmeb_classification(fxn, benchmark)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
