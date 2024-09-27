import os
import torch
import sys
import json

def load_saved_data(input_dir):

    # Load the dictionaries from JSON files
    with open(os.path.join(input_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)

    with open(os.path.join(input_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Load the tensors from binary files
    query = torch.load(os.path.join(input_dir, "query.pt"))
    cand = torch.load(os.path.join(input_dir, "cand.pt"))

    return dataset_info, metadata, query.cuda(), cand.cuda()

def compute_topk():
    with torch.no_grad():
        score_list = []
        top_k_list = []
        idx = 0
        top_k = int(sys.argv[2])
        path = sys.argv[1]
        dataset_info, metadata, query, cand = load_saved_data(path)
        batch_size = 100
        #assert(query.shape[0] == cand.shape[0])
        
        while idx < query.shape[0]:
            # will out of array if !(cand_size%batch_size==0)
            q_slice = query[idx:idx+batch_size,:]
            #100 x emb @ 300k x emb = 100x300k
            scores = q_slice @ cand.t()
            score, topk_idx = torch.topk(scores,top_k, dim=-1,) 
            score_list.append(score)
            top_k_list.append(topk_idx)
            idx += batch_size
   
        scores = torch.cat(top_k_list, dim=0)
        top_k = torch.cat(score_list, dim=0)
        out = torch.stack((top_k, scores))
        output_path = os.path.join(path, "top_k.pt")
        torch.save(out, output_path)

if __name__ == "__main__":
    compute_topk()
    print("done")