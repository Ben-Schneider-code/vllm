import os
from torch.utils.data import Dataset
import orjson

class WikiInstructAdapter(Dataset):
    
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
        self.meta = list(filter(lambda x: "parsed_text" in x, self.meta))
        

    def __len__(self):
        return len(self.meta)
        
    def _format_query(self):
        return None
    
    def _format_candidate(self):
        return None

    def _get_formatted_items(self, idx):
        item = self.meta[idx]
        obj = item["parsed_text"]



        return item


    # Currently the modality is image -> text
    def __getitem__(self, idx):
        meta = self.meta[idx]
        image = os.path.join(self.root, "images", f"{meta["id"]}.jpeg")
        assert os.path.exists(image)


        return None
    

def main():
    print("hello")

if __name__ == "__main__":
    main()