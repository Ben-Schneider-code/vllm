# ------------
# Evals against flickr30k
# ------------

import os
import sys
from tqdm import tqdm
import json
from evaluate.embed_function import get_model_with_embed_function

supported_models = ["abcQwenVL"]

import torch

def intersect(l1, l2):
    return len(set.intersection(set(l1), set(l2))) > 0

def get_topk_candidates(queries, candidates, k=3):
    """
    Return a dictionary where each key is a query id, and the value is a list 
    of the top-k candidate ids based on the dot-product similarity of embeddings.

    This version uses a vectorized approach with matrix multiplication.
    
    Args:
        queries (List[Tuple[str, torch.Tensor]]): 
            A list of (query_id, query_embedding).
        candidates (List[Tuple[str, torch.Tensor]]): 
            A list of (candidate_id, candidate_embedding).
        k (int): Number of top candidates to retrieve for each query.

    Returns:
        Dict[str, List[str]]: Mapping from query_id to list of top-k candidate_ids.
    """

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

def eval_flickr(fxn):
    
    mscoco_eval_path = os.environ.get("FLICKR_EVAL", None)
    assert(mscoco_eval_path is not None)
    with open(mscoco_eval_path, "rb") as f:
        mscoco_json = json.loads(f.read())["images"]

    test = list(filter(lambda x : x["split"] == "test", mscoco_json))

    filepath = "flickr30k_images"
    for x in test:
        x["image"] = os.path.join( os.path.dirname(mscoco_eval_path),filepath,x["filename"])
    
    text = []
    for _, x in enumerate(test):
        text.extend(x["sentences"])

    images = [(i["image"],fxn(i["image"], dtype="image")) for i in tqdm(test)]
    text = [(i["raw"],fxn(i["raw"], dtype="text")) for i in tqdm(text)]

    # i2t
    for topk in [1,5,10]:
        cand = get_topk_candidates(images, text, topk)
        acc = 0
        for x in test:
            targets = [i["raw"] for i in x["sentences"]]
            preds = cand[x["image"]]
            if intersect(preds, targets): acc += 1
        acc = acc / len(test)
        print(f"i2t top{topk} is {acc:.3f}")

    # t2i
    for topk in [1,5,10]:
        cand = get_topk_candidates(text, images, topk)
        acc = 0
        cntr = 0
        # each image has multiple associated text descriptions
        for x in test:
            for t in x["sentences"]:
                targets = [x["image"]]
                preds = cand[t["raw"]]
                if intersect(preds, targets): acc += 1
                cntr += 1
        acc = acc / cntr
        print(f"t2i top{topk} is {acc:.3f}")

def main(model_type: str, model_path: str):
    eval_flickr(load(model_type, model_path))

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
