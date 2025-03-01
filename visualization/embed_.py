import torch
from transformers import HfArgumentParser
from transformers.trainer_utils import PredictionOutput
from model.modeling_abc import MODEL_ARCHITECTURE
from internvl.patch.pad_data_collator import contrastive_data_collator
from util.contrastive_trainer import ContrastiveTrainer
from internvl.train.internvl_chat_finetune import VLMTrainingArguments, DataTrainingArguments, ModelArguments, build_contrastive_dataset, build_eval_datasets, load_model, setup_logger
import os
import sys
from monkey_patch.qwen_attn_patch import monkey_patch_transformers_lib, unmask_attn_monkey_patch
from torch import nn
from peft import PeftModel
from utils import save
from dataclasses import asdict


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
    output_paths = []
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    assert training_args.deepspeed is None, "Embedding does not support deepspeed"

    monkey_patch_transformers_lib()

    if MODEL_ARCHITECTURE[model_args.model_architecture].attn_mask == 'bidirectional':
        unmask_attn_monkey_patch()
    elif MODEL_ARCHITECTURE[model_args.model_architecture].attn_mask != 'casual':
        raise Exception("NotImplementedError")
        
    logger = setup_logger(training_args)
    model, tokenizer, tcs_loader = load_model(model_args, data_args, training_args, logger)
    
    model = merge_peft_submodules(model)

    for dataset_name in data_args.eval_datasets:
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
            "dataset_name": dataset_name,
            "model_args": asdict(model_args),
            "training_args": asdict(training_args),
            "data_args": asdict(data_args)
        }
        path = os.path.join(training_args.output_dir, dataset_name)
        save(dataset_info, meta,q,c,path)
        output_paths.append(path)
        print(output.metrics)
    return output_paths

# params - config_path and top_k
if __name__ == "__main__":
    output_paths = internvl_embed_dataset()

    neg_mine = int(sys.argv[-1])

    if neg_mine > 0:
        from visualization.neg_mine import compute_topk
        for path in output_paths:
            compute_topk(path, neg_mine)