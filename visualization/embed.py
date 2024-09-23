from transformers import HfArgumentParser
from transformers.trainer_utils import PredictionOutput
from internvl.patch.pad_data_collator import contrastive_data_collator
from internvl.train.contrastive_trainer import ContrastiveTrainer
from internvl.train.internvl_chat_finetune import VLMTrainingArguments, DataTrainingArguments, ModelArguments, build_contrastive_dataset, build_eval_datasets, load_model, setup_logger
import os
import sys

def embed_ds():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    logger = setup_logger(training_args)
    model, tokenizer, tcs_loader = load_model(model_args, data_args, training_args, logger)
    
    # switch to other ds builder when embedding full datasets
    dataset = build_eval_datasets(
    4,
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
    )
    dataset = dataset[list(dataset.keys())[0]]
    
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
    print(output)

if __name__ == "__main__":
    embed_ds()