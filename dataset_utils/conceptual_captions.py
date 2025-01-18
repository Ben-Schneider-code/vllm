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

class ConceptualCaptionsNegativeAdapter(ConceptualCaptionsAdapter):
    
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
                        "value": metadata["caption"] 

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
                        "value": ""
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

class VGInstructAdapter(Dataset):

    def __init__(self):
        assert "VG_ROOT" in os.environ, "Dataset location was not specified by env variable"
        self.root = os.environ["VG_ROOT"]
        


class ConceptualCaptionsInstructionAdapter(ConceptualCaptionsPretrainAdapter):

    def __init__(self, negatives=None):
        super().__init__(negatives=negatives)
        with open(os.path.join(self.root, "train_instructions.json"), 'rb') as f:
            self.instructions = orjson.loads(f.read())

        # Reduce dataset to only items we have instructions for
        keys = self.instructions.keys()
        self.instruction_meta = list(filter(lambda i: str(i["id"]) in keys, self.meta))
    
    def __len__(self):
        return len(self.instruction_meta)

    # Currently the modality is image -> text
    def __getitem__(self, idx):
        metadata = self.instruction_meta[idx]
        image = metadata["image"]
        id = str(metadata["id"])

        prompt_1 = self.instructions[id]["data"]["prompt 1"]
        caption_1 = self.instructions[id]["data"]["caption 1"]
        prompt_2 = self.instructions[id]["data"]["prompt 2"]
        caption_2 = self.instructions[id]["data"]["caption 2"]
        url = self.instructions[id]["url"]

        assert url == metadata["url"]
        formatted_item = [{
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
        },
        {
            "id": metadata["id"],
            "url": metadata["url"], 
            "pos_cand": self._create_candidate(metadata, custom_caption=caption_1),
            "query": {
                "id": metadata["id"],
                "image": image,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Instruction: {prompt_1}"
                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            }
        },
        {
            "id": metadata["id"],
            "url": metadata["url"], 
            "pos_cand":self._create_candidate(metadata, custom_caption=caption_2),
            "query": {
                "id": metadata["id"],
                "image": image,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Instruction: {prompt_2}"
                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            }
        }]
        


        if self.negatives is not None:
            self._attach_negatives(idx,formatted_item[0])

        return formatted_item

class CC128kAdapter(ConceptualCaptionsAdapter):

    def __init__(self):      
        assert "CC_ROOT" in os.environ, "Environment variable 'CC_ROOT' is not set"
        self.root = os.environ["CC_ROOT"]
        with open(os.path.join(self.root, "meta128k.json"), 'rb') as f:
            self.meta = orjson.loads(f.read())