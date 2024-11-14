import torch
from transformers import HfArgumentParser
from transformers.trainer_utils import PredictionOutput
from internvl.model.abc.modeling_abc import MODEL_ARCHITECTURE
from internvl.patch.pad_data_collator import contrastive_data_collator
from internvl.train.contrastive_trainer import ContrastiveTrainer
from internvl.train.internvl_chat_finetune import VLMTrainingArguments, DataTrainingArguments, ModelArguments, build_contrastive_dataset, build_eval_datasets, load_model, setup_logger
import os
import sys
from monkey_patch.qwen_attn_patch import forward_memory_opt_monkey_patch, unmask_attn_monkey_patch
from torch import nn
from peft import PeftModel
from utils import save
from dataclasses import asdict

def internvl_embed_dataset():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    dataset_name = data_args.training_dataset_name
    dataset = build_contrastive_dataset(
    data_args,
    None,
    None,
    None,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
    dataset_name = dataset_name
    )

    negative_path = os.path.join(dataset.adapter.root, "negative")
    c = torch.load(os.path.join(negative_path, "cand.pt"))
    q = torch.load(os.path.join(negative_path, "query.pt"))

    