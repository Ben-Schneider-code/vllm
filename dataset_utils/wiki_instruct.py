import os
from torch.utils.data import Dataset
import orjson

class WikiInstructAdapter(Dataset):
    
    prompt_keys = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4"]
    answer_keys = ["Answer1", "Answer2", "Answer3", "Answer4"]

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
        assert "WIKI_INSTRUCT_ROOT" in os.environ, "Environment variable 'WIKI_INSTRUCT_ROOT' is not set"
        self.root = os.environ["WIKI_INSTRUCT_ROOT"]
        with open(self.root, 'rb') as f:
            self.meta = orjson.loads(f.read())
        self.meta = list(filter(lambda x: ("parsed_text" in x) and os.path.exists(get_image_path(self.root, x["id"])), self.meta))
        
    def __len__(self):
        return len(self.meta)

    def _get_formatted_item(self, idx, url, image_path, instruction, answer):
        return {
            "id": idx,
            "url": url, 
            
            "query": {
                "id": idx,
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "",
                        #"value": f"Instruction: {instruction}"
                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            },
            "pos_cand": {
                "id": idx,
                "conversations": [
                    {
                        "from": "human",
                        "value": answer 

                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]
            }
        }

    def __getitem__(self, idx):
        meta = self.meta[idx]
        url = meta["url"]
        #image = os.path.join(os.path.dirname(self.root), "images", f"{meta["id"]}.jpeg")
        image = get_image_path(self.root, meta["id"])
        assert os.path.exists(image), f"Could not locate image, looked in {image}"
        parsed_text = meta["parsed_text"]

        ret = []

        for q, a in zip(self.prompt_keys, self.answer_keys):
            prompt = parsed_text[q]
            answer = parsed_text[a]
            ret.append(self._get_formatted_item(idx, url, image, prompt, answer))

        return ret

def get_image_path(root, image_id):
    os.path.join(os.path.dirname(root), "images", f"{image_id}.jpeg")