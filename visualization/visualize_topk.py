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
    top_k = torch.load(os.path.join(input_dir, "top_k.pt"))

    return dataset_info, metadata, query, cand, top_k

def display_img(path):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.show()

def get_cand_path(pth):
    return pth

def visualize():
    path=sys.argv[1]
    dataset_info, metadata, query, cand, top_k = load_saved_data(path)
    print("hello")
    



if __name__ == "__main__":
    visualize()
