import os
import torch
import sys

cand_and_query_are_symmetric = True

def load_saved_data(input_dir):

    # Load the dictionaries from JSON files
    #with open(os.path.join(input_dir, "dataset_info.json"), "r") as f:
    dataset_info = None # json.load(f)

    #with open(os.path.join(input_dir, "metadata.json"), "r") as f:
    metadata = None # json.load(f)

    # float32 - [680069, 896]
    query = torch.load(os.path.join(input_dir, "query.pt"))
    cand = torch.load(os.path.join(input_dir, "cand.pt"))
    return dataset_info, metadata, query.cuda(), cand.cuda()

def batch(query, cand, idx, batch_size, top_k):

    batch_idx = idx+batch_size if idx+batch_size < query.shape[0] else query.shape[0]
    q_slice = query[idx:batch_idx,:]
    #100 x emb @ 300k x emb = 100x300k
    scores = q_slice @ cand.t()
    
    
    score, topk_idx = torch.topk(scores,top_k, dim=-1,)
    

    c_slice = cand[idx:batch_idx,:].unsqueeze(-1)
    q_slice_view = q_slice.unsqueeze(1)
    scores_for_answer = torch.bmm(q_slice_view, c_slice)
    scores_for_answer = scores_for_answer.squeeze(dim=-1)

    mask = (scores > scores_for_answer)
    mask_sum = mask.sum(dim=-1)
    mask_sum_norm = mask_sum / cand.shape[0]
    print(torch.mean(mask_sum_norm))

    return score.detach().cpu(), topk_idx.detach().cpu(), mask_sum_norm.detach().cpu(), scores_for_answer.squeeze(dim=-1).detach().cpu()

def compute_topk(path, top_k):

    with torch.no_grad():
        score_list = []
        top_k_list = []

        # the proportion of candidates that the positive (correct) candidate had a higher score than
        relative_score = []
        # the score of the correct candidate
        scores_of_answer = []
        
        idx = 0
        _, _, query, cand = load_saved_data(path)
        batch_size = 3_000

        print("NUMBER OF QUERIES TO PROCESS " + str(query.shape[0]))
        while idx < query.shape[0]:
            s, t, p, a = batch(query,cand,idx,batch_size,top_k)
            score_list.append(s)
            top_k_list.append(t)
            relative_score.append(p)
            scores_of_answer.append(a)
            idx += batch_size


        scores_tensor = torch.cat(top_k_list, dim=0)
        top_k_tensor = torch.cat(score_list, dim=0)
        relative_score = torch.cat(relative_score, dim=0)
        scores_of_answer = torch.cat(scores_of_answer, dim=0)
        out = torch.stack((top_k_tensor, scores_tensor))
        
        torch.save(out, os.path.join(path, "top_k.pt"))
        torch.save(relative_score, os.path.join(path, "relative_scoring.pt"))
        torch.save(scores_of_answer, os.path.join(path,"absolute_scoring.pt"))

if __name__ == "__main__":
    top_k = int(sys.argv[2])
    path = sys.argv[1]

    compute_topk(path, top_k)
    print("done")
