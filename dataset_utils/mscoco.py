import os
import orjson
from torch.utils.data import Dataset, Subset
from PIL import Image
from functools import partial

class MSCOCO(Dataset):
    """
    Set your path to the dataset using the MSCOCO_ROOT env variable
    Excepected dataset structure is annotations, train2014, val2014
    """
    def __init__(self, train=True):
        self.root : str = os.environ["MSCOCO_ROOT"]
        self.train : bool = train
        self.image_path : str = os.path.join(self.root, "train2014") if train else os.path.join(self.root, "val2014")
        

        self.annotation_path : str = os.path.join(self.root, "annotations")
        self.annotation_path : str = os.path.join(self.annotation_path, "captions_train2014.json") if train else os.path.join(self.annotation_path, "captions_val2014.json")
        with open(self.annotation_path) as f:
            self.data_json = orjson.loads(f.read())
        
        self.data = self.data_json["annotations"]
        self.images = {element["id"] : element  for element in self.data_json["images"]}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        text_item = self.data[ind]
        image_item = self.images[text_item['image_id']]

        return {
            "text" : text_item["caption"],
            "image" : os.path.join(self.image_path, image_item["file_name"]),
            "text_id": text_item["id"],
            "image_id": image_item["id"],
            "url" : image_item["coco_url"],
            "id": str(ind)
        }
    
class MSCOCOAdapter(Dataset):

    def __init__(self):      
        assert "MSCOCO_ROOT" in os.environ, "Environment variable 'CC_ROOT' is not set"
        self.base_ds = MSCOCO()
        self.root=self.base_ds.root

    def __len__(self):
        return len(self.base_ds)
    
    # Currently the modality is image -> text
    def __getitem__(self, idx):
        metadata = self.base_ds[idx]

        formatted_item = {
            "id": metadata["id"],
            "url": metadata["url"], 
            "query": {
                "id": metadata["text_id"],
                "conversations": [
                    {
                        "from": "human",
                        "value": "Instruction: What kind of image would this caption be used for? Caption: " + metadata["text"] 

                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            },
            "pos_cand": {
                "id": metadata["image_id"],
                "image": metadata["image"],
                "conversations": [
                    {
                        "from": "human",
                        "value": "Describe this image in detail."
                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            }
        }

        return formatted_item

def CLIP_collate_fn(batch, processor=None):

    text_list = [t["text"] for t in batch]
    path_list = [i["image"] for i in batch]

    meta = [{
        "qid":batch[i]["text_id"],
        "pid":batch[i]["image_id"],
        "q_image":None,
        "p_image":batch[i]["image"],
        "q_conversation":batch[i]["text"],
        "p_conversation":None,
    } for i in range(len(text_list))]


    img_list = list(map(lambda x: Image.open(x), path_list))
    inputs = processor(text=text_list, images=img_list, return_tensors="pt", padding=True)
    return inputs, meta

def get_CLIP_collate_fn(processor):
    return partial(CLIP_collate_fn, processor=processor)

def get_first_n(n, train=True):
    return Subset(MSCOCO(train=train), range(n))