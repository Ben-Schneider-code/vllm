import torch
from typing import Dict
from torch.utils.data import Dataset, Subset
from qwen.vision_process import process_vision_info
from dataset_utils.conceptual_captions import CC128kAdapter, ConceptualCaptionsAdapter, ConceptualCaptionsNegativeAdapter, ConceptualCaptionsPretrainAdapter
from dataset_utils.mscoco import MSCOCOAdapter, MSCOCOInstructAdapter, MSCOCONegativeAdapter, MSCOCOPretrainAdapter
from dataset_utils.wiki_instruct import WikiInstructAdapter
from util.dataclass import DataTrainingArguments
import os
import random

class QwenCollate:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):

        # flatten list of lists
        if isinstance(features[0], list):
            features = [item for sublist in features for item in sublist]
        
        query_batch = [item["query_tokenized"] for item in features]
        cand_batch = [item["cand_tokenized"] for item in features]

 

        if 'negatives_tokenized' in features[0]:
            negatives_flattened = flatten_negatives(features)
            cand_batch.extend(negatives_flattened)
        
        query_text, query_img = unzip(query_batch)
        cand_text, _ = unzip(cand_batch)

        # attach metadata to batch
        meta = [{
                "qid" : item["query"]["id"],
                "q_image" : item["query"]["image"]  if "image" in item["query"] else None,
                "pid" : item["pos_cand"]["id"],
                "p_image" : item["pos_cand"]["image"] if "image" in item["pos_cand"] else None,
                "q_conversation" : item["query"]["conversations"],
                "p_conversation" : item["pos_cand"]["conversations"],
                } for item in features]

        query_batch =  self.processor(
        text=query_text,
        images=query_img,
        padding=True,
        return_tensors="pt",
        )

        cand_batch = self.processor(
        text=cand_text,
        padding=True,
        return_tensors="pt",
        )

        return {
            "query" : query_batch,
            "pos_cand": cand_batch,
            "meta": meta,
        }

def flatten_negatives(xss):
    return [x for xs in xss for x in xs["negatives_tokenized"]]

def unzip(tuples_list):
    list1, list2 = zip(*tuples_list)
    return list(list1), list(list2)

class QwenContrastiveDataset(Dataset):
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

    keyword_map = {
        "from": "role",
        "value": "content", # content is a list
        "gpt": "assistant",
        "human": "user"
    }


    def __init__(
        self,
        adapter,
        tokenizer,

    ):
        self.processor = tokenizer
        self.adapter = adapter
        self.root = adapter.root

    
    def __len__(self):
        return len(self.adapter)

    def process(self, conversation,image=None):
        """
        The adapter returns different conversation keys than the tokenizer requires.
        HF tokenizer uses "role": "user" or "assistant" and "content": <text>
        Qwen also requires that images are inserted between <img> tags, see:
        https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
        """

        formatted_conversation = []
        for turn in conversation:
            formatted_conversation.append(
                 {
                    self.keyword_map["from"] : self.keyword_map[turn["from"]],
                    self.keyword_map["value"]: [{"type": "text", "text": turn["value"]}]
                 }
            )

        if image is not None:
             formatted_conversation[0]["content"].insert(0, {
                  "type": "image",
                  "image": os.path.join(self.root, image)  
             })
        
        return formatted_conversation

    def tokenize_input(self, messages):
        
        if messages["conversations"][-1] == {"from" : "gpt", "value": ""}:
            del messages["conversations"][-1]

        formatted_conversation = self.process(messages["conversations"],image=messages.get("image"))

        text_input = self.processor.apply_chat_template(
            formatted_conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(formatted_conversation)
        
        if isinstance(image_inputs, list):
             image_inputs = image_inputs[0]

        return text_input, image_inputs

    def process_input(self, data_item):
        query = data_item["query"]
        cand = data_item["pos_cand"]
        data_item["query_tokenized"] = self.tokenize_input(query)
        data_item["cand_tokenized"] = self.tokenize_input(cand)
        if "negatives" in data_item:
            data_item["negatives_tokenized"] = [self.tokenize_input(n) for n in data_item["negatives"]]
        return data_item

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            data_item = self.adapter[i]
            if isinstance(data_item, dict):
                return self.process_input(data_item)
            elif isinstance(data_item, list):
                return [self.process_input(item) for item in data_item]
            else:
                raise Exception("InvalidTypeError")
        except:
            # if index fails just substitute in a random one
            return self.__getitem__(random.randint(0, self.__len__()))

def build_eval_datasets(
    eval_batch_size: int,
    data_args: DataTrainingArguments,
    tokenizer,

):

    if len(data_args.eval_datasets) == 0: return None
    
    subset_size = data_args.eval_steps_per_dataset*eval_batch_size
    eval_ds = {}

    for ds_name in data_args.eval_datasets:
        ds = build_contrastive_dataset(
        data_args,
        tokenizer,
        dataset_name=ds_name
        )
        indices = torch.randperm(len(ds))[:subset_size]
        eval_ds[ds_name] = Subset(ds, indices)

    return eval_ds

class Split(Dataset):
    def __init__(self, ds, idx) -> None:
        super().__init__()
        self.ds = ds
        self.idx = idx

    def __getitem__(self, i):
        return self.ds[self.idx[i]]

    def __len__(self):
        return len(self.idx)

# Last 128,000 samples are reserved for finetuning
def get_split(ds, pretrain=True):
    start_of_pretrain = 0
    end_of_finetune = len(ds)
    end_of_pretrain = end_of_finetune - 128000

    if pretrain:
        ds = Split(ds, list(range(start_of_pretrain, end_of_pretrain)))
    else:
        ds = Split(ds, list(range(end_of_pretrain, end_of_finetune)))

    return ds


def build_contrastive_dataset(
    data_args,
    tokenizer,
    dataset_name = None,
    is_train=False # whether to use negatives or not
):  

    if dataset_name == 'cc':
        dataset = QwenContrastiveDataset(
                ConceptualCaptionsAdapter(),
                tokenizer,
            )
    elif dataset_name == 'cc128k':
        dataset = QwenContrastiveDataset(
                CC128kAdapter(),
                tokenizer,
            )
    elif dataset_name == 'mscoco':
        dataset = QwenContrastiveDataset(
                MSCOCOAdapter(),
                tokenizer,
            )
    elif dataset_name == "cc_pretrain":
                dataset = QwenContrastiveDataset(
                get_split(ConceptualCaptionsPretrainAdapter(negatives=data_args.negatives if is_train else None), pretrain=True),
                tokenizer
            )
    elif dataset_name == "mscoco_pretrain":
            dataset = QwenContrastiveDataset(
            MSCOCOPretrainAdapter(negatives=data_args.negatives if is_train else None),
            tokenizer,

        )       
    elif dataset_name == "mscoco_instruct":
        dataset = QwenContrastiveDataset(
        MSCOCOInstructAdapter(),
        tokenizer,
    )
    else:
        raise Exception("NotImplementedError")
    
    return dataset