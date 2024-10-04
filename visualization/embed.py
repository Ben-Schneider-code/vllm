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
import json 
from torch import nn
from peft import PeftModel
import deepspeed
from utils import save

def merge_peft_submodules(module: nn.Module) -> nn.Module:
    """
    Recursively merge all PEFT submodules within a PyTorch module.
    
    Args:
        module (nn.Module): The PyTorch module to process.
    
    Returns:
        nn.Module: The module with all PEFT submodules merged.
    """
    for name, child in module.named_children():
        if isinstance(child, PeftModel):
            # Merge the PEFT model
            merged_model = child.merge_and_unload()
            setattr(module, name, merged_model)
        else:
            # Recursively process child modules
            merge_peft_submodules(child)
    
    return module


def internvl_embed_dataset():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
     
    forward_memory_opt_monkey_patch()
    
    if MODEL_ARCHITECTURE[model_args["model_architecture"]].attn_mask == 'bidirectional':
        unmask_attn_monkey_patch()
    elif MODEL_ARCHITECTURE[model_args["model_architecture"]].attn_mask != 'casual':
        raise Exception("NotImplementedError")
        
    logger = setup_logger(training_args)
    model, tokenizer, tcs_loader = load_model(model_args, data_args, training_args, logger)
    
    with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0):
        model = merge_peft_submodules(model)
    dataset_name = data_args.eval_datasets[0]   

    dataset = build_contrastive_dataset(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
    dataset_name = dataset_name
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=contrastive_data_collator,
        wandb=False
    )

    trainer.prediction_step = trainer.embed_step
    output: PredictionOutput  = trainer.predict(dataset)

    preds = output.predictions
    
    meta = []
    q = []
    c = []

    for i in range(len(preds)):
        meta.extend(preds[i]["meta"])
        q.extend(preds[i]["q"])
        c.extend(preds[i]["c"])

    q = torch.stack(q, dim=0)
    c = torch.stack(c, dim=0)
    
    dataset_info = {
        "model_name": model_args.model_name_or_path,
        "dataset_name": dataset_name
    }
    
    save(dataset_info, meta,q,c,training_args.output_dir)
    print(output.metrics)

if __name__ == "__main__":
    internvl_embed_dataset()