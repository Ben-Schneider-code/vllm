import torch
from transformers import HfArgumentParser
from transformers.trainer_utils import PredictionOutput
from internvl.patch.pad_data_collator import contrastive_data_collator
from internvl.train.contrastive_trainer import ContrastiveTrainer
from internvl.train.internvl_chat_finetune import VLMTrainingArguments, DataTrainingArguments, ModelArguments, build_contrastive_dataset, build_eval_datasets, load_model, setup_logger
import os
import sys
from monkey_patch.qwen_attn_patch import qwen_memory_opt, unmask_qwen2_attn
import json 

def save(dataset_info: dict,
        metadata: dict,
        query: torch.tensor,
        cand: torch.tensor,
        output_dir: str):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the dictionaries as JSON files
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # Save the tensors as binary files
    torch.save(query, os.path.join(output_dir, "query.pt"))
    torch.save(cand, os.path.join(output_dir, "cand.pt"))

    print(f"Data saved to '{output_dir}'.")

def internvl_embed_dataset():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
     
    qwen_memory_opt()
    
    if training_args.attn_mask == 'bidirectional':
        unmask_qwen2_attn()
        
    logger = setup_logger(training_args)
    model, tokenizer, tcs_loader = load_model(model_args, data_args, training_args, logger)

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

    from torch.utils.data import Subset
    dataset = Subset(dataset=dataset,indices=range(2000))
    
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=contrastive_data_collator
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

    save(dataset_info, meta, q,c,training_args.output_dir)

if __name__ == "__main__":
    internvl_embed_dataset()