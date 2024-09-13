import torch
import os
import orjson 

class ConceptualCaptionsAdapter(torch.utils.data.Dataset):
    
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
        self.dataset_root = os.environ["CC_ROOT"]
        with open(os.path.join(self.dataset_root, "meta.json"), 'rb') as f:
            self.meta = orjson.loads(f.read())
        

    def __len__(self):
        return len(self.meta)
    
    # Currently the modality is image -> text
    def __getitem__(self, idx):
        metadata = self.meta[idx]
        image = os.path.join(self.dataset_root, metadata["image"])
        # is it okay to not add any gpt msg?
        formatted_item = {
            "id": metadata["id"],
            "url": metadata["url"], 
            "query": {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Intruction: find an image that matches this caption. Caption: " + metadata["caption"] 
                    }
                ]
            },
            "pos_cand": {
                "image": image,
                "conversations": [
                    {
                        "from": "human",
                        "value": "Describe this image."
                    }
                ]
            }
        }

        return formatted_item