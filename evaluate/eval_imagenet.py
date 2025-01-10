# ------------
# Evals against mscoco
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

def eval_imagenet(fxn):
    
    imagenet_eval_path = os.environ.get("IMAGENET_EVAL", None)
    assert(imagenet_eval_path is not None)
    
    with open(os.path.join(imagenet_eval_path, "jpeg_map.json"), "r") as file:
        file_to_class_id = json.load(file) 

    with open(os.path.join(imagenet_eval_path, "LOC_synset_mapping.txt"), "r") as file:
        id_to_plaintext = {}
        for line in file:
            parts = line.strip().split(" ", 1)  # Split into ID and name
            id_to_plaintext[parts[0]] = parts[1]
    
    image_path = os.path.join(imagenet_eval_path, "val")
    img_files = [f"ILSVRC2012_val_{str(number).zfill(8)}.JPEG" for number in range(1,50_001)]

    l = []
    for file_name in img_files:
        class_id = file_to_class_id[file_name]
        class_plaintext = id_to_plaintext[class_id]
        l.append((file_name, class_id, class_plaintext))
    
    assert(len(l) == 50000)

    import random
    random.seed(10)
    l = random.choices(l, k=5000)

    images = [(i[0],fxn(os.path.join(image_path,i[0]), dtype="image")) for i in tqdm(l)]
    text = [(i,fxn(f"A photo of {id_to_plaintext[i]}.", dtype="text")) for i in tqdm(id_to_plaintext.keys())]

    # i2t
    for topk in [1,5,10]:
        cand = get_topk_candidates(images, text, topk)
        acc = 0
        for x in l:
            targets = [x[1]]
            preds = cand[x[0]]
            if intersect(preds, targets): acc += 1
        acc = acc / len(l)
        print(f"i2t top{topk} is {acc:.4f}")

def main(model_type: str, model_path: str):
    eval_imagenet(load(model_type, model_path))

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
