import os
import orjson
from torch.utils.data import Dataset
import requests
from PIL import Image
from io import BytesIO

class InstructionFiltering(Dataset):


    def __init__(self, prompt) -> None:
        super().__init__()
        self.prompt = prompt
        self.item_idx = []
    
        with open(os.environ["FILTERED_WIKIWEB"], "rb") as file:
            self.data = orjson.loads(file.read())

        for ind, element in enumerate(self.data):
            l =list(element["section_image_url"])
            self.item_idx.extend(list(zip([ind]*len(l), l)))

    def __getitem__(self, idx):
        data_idx, image_idx = self.item_idx[idx]
        data_item = self.data[data_idx]
        image_url = data_item["section_image_url"][image_idx]
        section_idx = int(image_idx.split('_', 1)[0])
        
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
            image = None

        return image, "<image>"+self.prompt+'\n'+data_item["section_text"][section_idx]

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
    