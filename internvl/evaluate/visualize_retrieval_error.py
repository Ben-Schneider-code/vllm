from typing import Dict, Union, Any, Optional, List, Tuple
import torch
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, TrainingArguments,
                          set_seed)
import sys
import os
from transformers.trainer_utils import get_last_checkpoint
from internvl.train.internvl_chat_finetune import (DataTrainingArguments,
                                                    ModelArguments,
                                                    setup_logger,
                                                    load_model,
                                                    build_contrastive_dataset,
                                                    )
from internvl.train.contrastive_trainer import ContrastiveTrainer
from internvl.patch import contrastive_data_collator
from torch.utils.data import Subset
import torch.nn.functional as F

def compute_contrastive_loss(q_embeds, p_embeds):  # [batch_size, embed_dim]
    # Normalized features
    q_embeds = F.normalize(q_embeds, dim=-1)
    p_embeds = F.normalize(p_embeds, dim=-1)
    bs = q_embeds.size(0)

    score = torch.matmul(q_embeds, p_embeds.t())  # * self.logit_scale  # [bs, bs]
    sim_targets = torch.arange(bs).to(score.device)  # [bs]

    # compute loss
    loss = F.cross_entropy(score, sim_targets)
    _max_score, max_idxs = torch.max(score, 1)
    topk_values, topk_indices = torch.topk(score, k=5, dim=1)

    accuracy = (max_idxs == sim_targets).sum() / bs

    return loss, accuracy, topk_indices

def compute_loss_with_visualization(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    query = inputs["query"]
    candidate = inputs["pos_cand"]

    query_token_id = self.tokenizer.convert_tokens_to_ids("<CLS_1>")
    cand_token_id = self.tokenizer.convert_tokens_to_ids("<CLS_2>")

    query_mask = (query["input_ids"] == query_token_id).to(torch.long)
    cand_mask = (candidate["input_ids"] == cand_token_id).to(torch.long)

    # found exactly BATCH_SIZE special tokens.
    assert(torch.sum(query_mask) == query_mask.shape[0])
    assert(torch.sum(cand_mask) == cand_mask.shape[0])

    query_token_idx = torch.argmax(query_mask, dim=1)
    cand_token_idx = torch.argmax(cand_mask, dim=1)

    query.pop("labels")
    candidate.pop("labels")

    # ensure that sequences are **not** padded to context length
    query_outputs = model(**query, output_hidden_states=True)
    candidate_outputs = model(**candidate, output_hidden_states=True)
    
    assert(query_outputs.logits is None)
    assert(candidate_outputs.logits is None)

    batch_idx = torch.arange(query["input_ids"].shape[0])
    
    q_embed = query_outputs.hidden_states[-1][batch_idx,query_token_idx]
    c_embed = candidate_outputs.hidden_states[-1][batch_idx,cand_token_idx]
    
    q_embed_mlp = model.module.mlp_q(q_embed)
    c_embed_mlp = model.module.mlp_c(c_embed)

    q_embed_mlp_float = q_embed_mlp.float()
    c_embed_mlp_float = c_embed_mlp.float()

    # Concatenate the gathered embeddings along the batch dimension
    q_embed_gathered = q_embed_mlp_float #torch.cat(q_embed_list, dim=0)
    c_embed_gathered = c_embed_mlp_float #torch.cat(c_embed_list, dim=0)

    loss, acc, top_k = compute_contrastive_loss(q_embed_gathered, c_embed_gathered)

    return loss

def prediction_step(
    self,
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    prediction_loss_only: bool,
    ignore_keys: Optional[List[str]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

    inputs = self._prepare_inputs(inputs)
   
    with torch.no_grad():
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.mean().detach()

    return (loss, None, None)

ContrastiveTrainer.prediction_step = prediction_step
ContrastiveTrainer.compute_loss = compute_loss_with_visualization

def visualize():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))

    logger = setup_logger(training_args)
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    model, tokenizer, tcs_loader = load_model(model_args, data_args, training_args, logger)

    train_dataset = build_contrastive_dataset(
    data_args,
    tokenizer,
    tcs_loader,
    model
    )

    dataset_size = len(train_dataset)
    indices = torch.randperm(dataset_size).tolist()  # Generate a random permutation of indices
    subset_indices = indices[:1000]  # Select the first 1000 indices
    eval_subset = Subset(train_dataset, subset_indices)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=contrastive_data_collator
    )

    trainer.predict(
        test_dataset=eval_subset
    )

visualize()