import torch
import torch.distributed as dist
import os, sys, logging
from transformers import set_seed, HfArgumentParser
from util.contrastive_trainer import ContrastiveTrainer
from util.dataclass import ModelArguments, DataTrainingArguments, VLMTrainingArguments
from transformers import Qwen2VLModel, Qwen2VLProcessor
from qwen.qwen_dataset import build_contrastive_dataset, build_eval_datasets, QwenCollate

logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def load_model(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: VLMTrainingArguments):

    min_pixels = 256*28*28
    max_pixels = 512*28*28
    
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path,
                                                padding_side="right",
                                                use_fast=False,
                                                max_pixels=max_pixels)

    model = Qwen2VLModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    return model, processor

def init_instruction_finetuning():
    return None

def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    model, tokenizer = load_model(model_args, data_args, training_args)

    # if we are doing instruction finetuning fuse the LoRA weights and init new ones
    if model_args.instruction_mode:
        model = init_instruction_finetuning(model)

    train_dataset = build_contrastive_dataset(
    data_args,
    tokenizer,
    dataset_name=data_args.training_dataset_name,
    is_train=True
    )  

    eval_dataset = build_eval_datasets(
        training_args.per_device_eval_batch_size,
        data_args,
        tokenizer
    )

    from torch.utils.data import DataLoader
    dl = DataLoader(dataset=train_dataset, batch_size=8, num_workers=0, collate_fn=QwenCollate(tokenizer))
    batch = next(enumerate(dl))
    exit()
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    has_lora_weights = [key for key in model.state_dict().keys() if 'lora' in key.lower()]
    if has_lora_weights: print("Has lora weight already, skipping lora init")
    if model_args.use_backbone_lora and not has_lora_weights:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora,
                                lora_alpha=2*model_args.use_backbone_lora,
                                lora_dropout=model_args.lora_dropout,
                                use_dora=model_args.use_dora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora and not has_lora_weights:
        model.wrap_llm_lora(r=model_args.use_llm_lora,
                            lora_alpha=2*model_args.use_llm_lora,
                            lora_dropout=model_args.lora_dropout,
                            use_dora=model_args.use_dora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        #data_collator=contrastive_data_collator
    )

    # # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()  # Saves the tokenizer too for easy upload

    #     metrics = train_result.metrics
    #     try:
    #         metrics['train_samples'] = len(train_dataset)
    #     except:
    #         metrics['train_samples'] = -1

    #     trainer.log_metrics('train', metrics)
    #     trainer.save_metrics('train', metrics)
    #     trainer.save_state()

if __name__ == '__main__':
    main()