import os
import orjson 
from torch.utils.data import Dataset
import numpy as np

class ConceptualCaptionsAdapter(Dataset):
    
    """
    **An adapter must return a data element in the following format**
    {
            "query": {
                optional<"image": str_path>,
                "id": optional<any>,
                "conversations": [
                    {
                        "from": "human",
                        "value": str
                    },
                    {
                        "from": "gpt",
                        "value": str
                    }
                ]
            },

            "pos_cand": {
                optional<"image": str_path>,
                "id": optional<any>,
                "conversations": [
                    {
                        "from": "human",
                        "value": str
                    },
                    {
                    "from": "gpt",
                    "value": str
                    }
                ]
            }
    }
    """

    def __init__(self):      
        assert "CC_ROOT" in os.environ, "Environment variable 'CC_ROOT' is not set"
        self.root = os.environ["CC_ROOT"]
        with open(os.path.join(self.root, "meta.json"), 'rb') as f:
            self.meta = orjson.loads(f.read())
        

    def __len__(self):
        return len(self.meta)
    
    # Currently the modality is image -> text
    def __getitem__(self, idx):
        metadata = self.meta[idx]
        image = metadata["image"]

        formatted_item = {
            "id": metadata["id"],
            "url": metadata["url"], 
            "query": {
                "id": metadata["id"],
                "conversations": [
                    {
                        "from": "human",
                        "value": "Instruction: What kind of image would this caption be used for? Caption: " + metadata["caption"] 

                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            },
            "pos_cand": {
                "id": metadata["id"],
                "image": image,
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
    
class ConceptualCaptionsPretrainAdapter(ConceptualCaptionsAdapter):

    def __init__(self, negatives=None):      
        assert "CC_PRETRAIN_ROOT" in os.environ, "Environment variable 'CC_PRETRAIN_ROOT' is not set"
        self.root = os.environ["CC_PRETRAIN_ROOT"]
        with open(os.path.join(self.root, "meta.json"), 'rb') as f:
            self.meta = orjson.loads(f.read())
        
        self.negatives = negatives
        if self.negatives is not None:
            with open(os.path.join(self.root, "negatives.json"), 'rb') as f:
                self.negative_meta = orjson.loads(f.read())
            assert len(self.meta) == len(self.negative_meta)


    def _attach_negatives(self, idx, item):
        
        offset = 5
        
        starts = np.arange(self.negatives) * offset
        offsets = np.random.randint(0, offset, size=self.negatives)
        neg_idx = starts + offsets

        negatives_for_idx = self.negative_meta[str(idx)]
        item["negatives"] = [self._create_candidate(self.meta[negatives_for_idx[i]]) for i in neg_idx]


    def _create_candidate(self, metadata, custom_caption = None):

        caption = metadata["caption"] 

        if custom_caption:
            caption = custom_caption

        return {
                "id": metadata["id"],
                "conversations": [
                    {
                        "from": "human",
                        "value": caption

                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            }

    
    # Currently the modality is image -> text
    def __getitem__(self, idx):
        metadata = self.meta[idx]
        image = metadata["image"]

        formatted_item = {
            "id": metadata["id"],
            "url": metadata["url"], 
            "pos_cand": self._create_candidate(metadata),
            "query": {
                "id": metadata["id"],
                "image": image,
                "conversations": [
                    {
                        "from": "human",
                        "value": ""
                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            }
        }
        
        if self.negatives is not None:
            self._attach_negatives(idx,formatted_item)

        return formatted_item

def format_cand(txt):
    return {
                "id": "nil",
                "conversations": [
                    {
                        "from": "human",
                        "value": txt

                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
    }

def format_inst_query(img_path, inst):
    return {
                "id": "nil",
                "image": img_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Instruction: {inst}"
                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            }

class VGInstructAdapter(Dataset):

    def __init__(self):
        assert "VG_ROOT" in os.environ, "Dataset location was not specified by env variable"
        self.root = os.environ["VG_ROOT"]
        with open(os.path.join(self.root, "dataset.json")) as f:
            self.ds_defintion = orjson.loads(f.read())

    def __len__(self):
        return len(self.ds_defintion)

    def __getitem__(self, idx):
        
        items = self.ds_defintion[idx]

        return [{
            "id": str(i["id"]),
            "url": "nil", 
            "pos_cand": format_cand(i["phrase"]),
            "query": format_inst_query(os.path.join(self.root, "VG_100K", f"{i["image"]}.jpg"), i["instruction"])
        } for i in items]