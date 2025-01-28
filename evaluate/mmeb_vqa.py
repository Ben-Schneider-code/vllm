# ------------
# Evals against the mmeb vqa set
# TODO THIS IS BROKEN
# ------------

import os
import sys
from tqdm import tqdm
from evaluate.embed_function import get_model_with_embed_function
from datasets import load_dataset
splits = ["OK-VQA", "A-OKVQA","DocVQA","InfographicsVQA","ChartQA","Visual7W","ScienceQA","VizWiz","GQA","TextVQA"]
import torch

supported_models = ["abcQwenVL-Instruct"]

def intersect(l1, l2):
    return len(set.intersection(set(l1), set(l2))) > 0

def get_topk_candidates(queries, candidates, k=3):

    query_ids, query_embs = zip(*queries)      
    candidate_ids, candidate_embs = zip(*candidates)
    query_stack = torch.cat(query_embs, dim=0).cuda()
    candidate_stack = torch.stack(candidate_embs, dim=0).cuda()
    scores = (query_stack @ candidate_stack.T).cpu()
    topk_values, topk_indices = torch.topk(scores, k=k, dim=1)

    # Build the results dictionary
    results = {}
    for i, q_id in enumerate(query_ids):
        # topk_indices[i] gives the indices of the top-k candidates for this query
        top_candidate_ids = [candidate_ids[idx] for idx in topk_indices[i]]
        results[q_id] = top_candidate_ids

    return results

def load(model_type, pretrain_model_path, instruct_model_path, batch=False):
    return get_model_with_embed_function(model_type, pretrain_model_path, instruct_model_path=instruct_model_path,batch=batch)

def unroll_split(ds):
    labels = []
    query = []

    for item in ds:
        target_list = item["tgt_text"]
        labels.append(target_list)
        instruction = item["qry_text"]
        assert instruction.startswith("<|image_1|>\n"), "String does not start with the expected prefix."
        
        # Remove the prefix
        instruction = instruction[len("<|image_1|>\n"):]
        query.append({"img": item["qry_img_path"], "instruction": instruction, "target": target_list[0]})

        # assert that we can reuse the same embeddings
    return query, labels

def eval_mmeb_classification(fxn, split_name):

    mmeb_path = os.environ["MMEB_EVAL"]
    ds = load_dataset("TIGER-Lab/MMEB-eval", split_name)["test"]
    q, c = unroll_split(ds)

    
    
    text = []
    for item in tqdm(c):
        batch = [f"The answer is {i}." for i in item]
        emb_batch = fxn(batch, dtype="text")
        text.append([(item, emb_batch[ind, :]) for (ind, item) in enumerate(item)])
    
    images = [(i["img"],fxn(os.path.join(mmeb_path,i["img"]), dtype="image", instruction=i["instruction"])) for i in tqdm(q)]

    # i2t
    print(f"{split_name}")
    for topk in [1]:
        acc = 0
        for ind in range(len(q)):
            query = q[ind]
            text_batch = text[ind]

            cand = get_topk_candidates(images, text_batch, topk)
            targets = [query["target"]]
            preds = cand[query["img"]]
            if intersect(preds, targets): acc += 1
        acc = acc / len(images)
        print(f"i2t top{topk} is {acc:.4f}")


def main(model_type: str, pretrain_model_path: str, instruct_model_path: str):
    fxn = load(model_type, pretrain_model_path, instruct_model_path, batch=True)
    for benchmark in splits:
        eval_mmeb_classification(fxn, benchmark)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
