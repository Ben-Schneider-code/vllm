import torch
from transformers import HfArgumentParser
from internvl.train.internvl_chat_finetune import VLMTrainingArguments, DataTrainingArguments, ModelArguments, build_contrastive_dataset
import os
import sys
from tqdm import tqdm
from math import inf
import json

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VLMTrainingArguments))
    _, data_args, _ = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    dataset_name = data_args.training_dataset_name
    dataset = build_contrastive_dataset(
    data_args,
    None,
    None,
    None,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
    dataset_name = dataset_name
    )

    negative_path: os.PathLike = os.path.join(dataset.adapter.root, "negative")
    c: torch.Tensor = torch.load(os.path.join(negative_path, "cand.pt")).cuda()
    q: torch.Tensor = torch.load(os.path.join(negative_path, "query.pt"))

    num_samples: int = q.shape[0]
    negative_dict: dict = {}

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Processing Samples"):
            negatives_for_sample(idx, negative_dict, q, c)
    
    output_path = os.path.join(negative_path, 'negatives.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(negative_dict, f)

def negatives_for_sample(idx: int, negative_dict: dict, q: torch.Tensor, c: torch.Tensor):
    query = q[idx,:].cuda().unsqueeze(dim=0)
    scores = (c @ query.t()).squeeze(dim=-1)
    score_threshold = scores[idx]*.95
    scores[scores>score_threshold] = -inf
    sorted_indices = torch.argsort(scores, descending=True)
    top_negatives = sorted_indices[:100].tolist()
    negative_dict[idx] = top_negatives
    
if __name__ == "__main__":
    main()