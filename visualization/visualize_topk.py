import os
import torch
import sys
import json
from pathlib import Path
import shutil

ds_path = {
    "cc": os.environ["CC_ROOT"],
    "mscoco": os.environ["MSCOCO_ROOT"]
}

def load_saved_data(input_dir):

    # Load the dictionaries from JSON files
    with open(os.path.join(input_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)

    with open(os.path.join(input_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Load the tensors from binary files
    query = torch.load(os.path.join(input_dir, "query.pt"))
    cand = torch.load(os.path.join(input_dir, "cand.pt"))
    top_k = torch.load(os.path.join(input_dir, "top_k.pt"))

    return dataset_info, metadata, query, cand, top_k

def get_img_path(dataset,item_path):
    return os.path.join(ds_path[dataset], item_path)

def visualize():
    path=sys.argv[1]
    dataset_info, metadata, query, cand, top_k = load_saved_data(path)
    low = int(sys.argv[2])
    high = int(sys.argv[3])
    
    pos = metadata[low:high]
    negative_indexes = top_k[1,low:high,:]
    
    for i in range(len(pos)):
        query_sring = pos[i]["q_conversation"][0]["value"]
        output_dict = {}
        neg_data = negative_indexes[i].tolist()
        neg_data = [metadata[int(i)] for i in neg_data]
        for idx, item in enumerate(neg_data):
            output_dict[str(idx)+ "_candidate.jpeg"] = get_img_path(dataset_info["dataset_name"], item["p_image"])
        output_dir = "./visual/"+str(i)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for key in output_dict:
            shutil.copyfile(output_dict[key], os.path.join(output_dir,key))
        with open(os.path.join(output_dir,"query.txt"), "w") as f:
            f.write(query_sring)    
if __name__ == "__main__":
    visualize()
