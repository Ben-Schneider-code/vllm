import torch
import os
import json

def save(dataset_info: dict,
        metadata: dict,
        query: torch.tensor,
        cand: torch.tensor,
        output_dir: str):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the dictionaries as JSON files
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # Save the tensors as binary files
    torch.save(query, os.path.join(output_dir, "query.pt"))
    torch.save(cand, os.path.join(output_dir, "cand.pt"))

    print(f"Data saved to '{output_dir}'.")