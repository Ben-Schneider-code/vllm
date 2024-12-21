import torch
from typing import Dict
from torch.utils.data import Dataset, Subset
from qwen.vision_process import process_vision_info
from dataset_utils.conceptual_captions import CC128kAdapter, ConceptualCaptionsAdapter, ConceptualCaptionsNegativeAdapter, ConceptualCaptionsPretrainAdapter
from dataset_utils.mscoco import MSCOCOAdapter, MSCOCOInstructAdapter, MSCOCONegativeAdapter, MSCOCOPretrainAdapter
from dataset_utils.wiki_instruct import WikiInstructAdapter
from util.dataclass import DataTrainingArguments

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

    def tokenize_input(self, messages):
        print(messages)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        ) 

    def process_input(self, data_item):
        query = data_item["query"]
        cand = data_item["pos_cand"]
        data_item["query_tokenized"] = self.tokenize_input(query)
        data_item["cand_tokenized"] = self.tokenize_input(cand)
        if "negatives" in data_item:
            data_item["negatives_tokenized"] = [self.tokenize_input(n) for n in data_item["negatives"]]
        return data_item

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_item = self.adapter[i]
        if isinstance(data_item, dict):
            return self.process_input(data_item)
        elif isinstance(data_item, list):
            return [self.process_input(item) for item in data_item]
        else:
            raise Exception("InvalidTypeError")
        

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
    elif dataset_name == 'cc_neg':
                dataset = QwenContrastiveDataset(
                ConceptualCaptionsNegativeAdapter(),
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
    elif dataset_name == 'mscoco_neg':
                dataset = QwenContrastiveDataset(
                MSCOCONegativeAdapter(),
                tokenizer
            )
    elif dataset_name == "cc_pretrain":
                dataset = QwenContrastiveDataset(
                ConceptualCaptionsPretrainAdapter(negatives=data_args.negatives if is_train else None),
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
    elif dataset_name == "wiki_instruct":
            dataset = QwenContrastiveDataset(
            WikiInstructAdapter(),
            tokenizer,
        )
    else:
        raise Exception("NotImplementedError")
    
    return dataset