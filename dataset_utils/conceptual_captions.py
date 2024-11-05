import os
import orjson 
from torch.utils.data import Dataset

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

class ConceptualCaptionsAdapterITT(Dataset):

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
            "pos_cand": {
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
            "query": {
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

class CC128kAdapter(ConceptualCaptionsAdapter):

    def __init__(self):      
        assert "CC_ROOT" in os.environ, "Environment variable 'CC_ROOT' is not set"
        self.root = os.environ["CC_ROOT"]
        with open(os.path.join(self.root, "meta128k.json"), 'rb') as f:
            self.meta = orjson.loads(f.read())