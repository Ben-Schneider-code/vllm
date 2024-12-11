import os
import orjson
from torch.utils.data import Dataset
import requests
from PIL import Image
from io import BytesIO


def run_qwen2_vl(question: str, modality: str):
    assert modality == "image"

    prompt = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    stop_token_ids = None
    return  prompt, stop_token_ids

class InstructionFiltering(Dataset):

    def __init__(self, prompt) -> None:
        super().__init__()
        self.prompt = prompt
        self.item_idx = []
        self. modality = "image"
        assert "FILTERED_WIKIWEB" in os.environ, "Please add a path to your filtered wikiweb dataset"
        with open(os.environ["FILTERED_WIKIWEB"], "rb") as file:
            self.data = orjson.loads(file.read())

        for ind, element in enumerate(self.data):
            l =list(element["section_image_url"])
            self.item_idx.extend(list(zip([ind]*len(l), l)))

    def __getitem__(self, idx):
        data_idx, image_idx = self.item_idx[idx]
        data_item = self.data[data_idx]
        image_url = data_item["section_image_url"][image_idx]
        #section_idx = int(image_idx.split('_', 1)[0])
        
        # Define a User-Agent header for Wikimedia requests
        headers = {
            "User-Agent": "University_of_Waterloo_DataMining/1.0 contact:benjamin.schneider@uwaterloo.ca)"
        }

        try:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()  # Check if the request was successful
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(e)
            return None

        user_prompt = "Instruction: " + self.prompt
        prompt_templated, _ = run_qwen2_vl(user_prompt, self.modality)

        meta = {
            "article_url": data_item["url"],
            "image_url": image_url,
            "section_text": data_item["section_text"]
        }

        return idx,meta, {
            "prompt": prompt_templated,
            "multi_modal_data": {
                self.modality: image.convert("RGB")
            },
        }

    def batch_attach(self, idx, batch):
        for i in idx: self.attach(i, batch)

    def attach(self, idx, item):
        data_idx, image_idx = self.item_idx[idx]
        data_item = self.data[data_idx]
        d = data_item.setdefault("response_plaintext", {})
        d[image_idx] = item
     
    def save(self):
        save_data_path = "instruct_wikiweb.json"
        with open(save_data_path, "wb") as output_file:
            output_file.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))
    
    def __len__(self):
        return len(self.item_idx)

def qwen_collator(batch):

    # Filter out failed requests
    batch = list(filter(lambda x: x is not None, batch))
    # Unpack into two lists
    ids, meta, data_batch = zip(*batch)
    meta = list(meta)
    ids = list(ids)
    data_batch = list(data_batch)
    return ids, meta, data_batch