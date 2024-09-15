from typing import Optional
import torch
from transformers import Trainer
from transformers.trainer import RandomSampler
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.distributed as dist
from transformers.integrations import WandbCallback
import os

class WandbLogger(WandbCallback):

    def __init__(self):
        super().__init__()
        self.additional_metrics = {}

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        if state.is_world_process_zero:
            self._wandb.config.update({"pid": str(os.getpid())}, allow_val_change=True)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model=model, logs=logs | self.additional_metrics, **kwargs)
        self.additional_metrics.clear()

class ContrastiveTrainer(Trainer):

     def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.wandb_callback = WandbLogger()
          self.add_callback(self.wandb_callback)

     def log_to_wandb(self, key, value):
          self.wandb_callback.additional_metrics[key] = value

     def last_token_loss(self, model, inputs, return_outputs=False):

          query = inputs["query"]
          candidate = inputs["pos_cand"]

          # torch.cuda.memory._record_memory_history(max_entries=40)

          # MEMORY OPTIMIZATIONS
          # --------------------
          # make sure neither of these have any "labels" key
          # skips the loss computation
          query.pop("labels")
          candidate.pop("labels")
          # ensure that sequences are **not** padded to contexzt length
          query_outputs : CausalLMOutputWithPast = model(**query, output_hidden_states=True)
          candidate_outputs : CausalLMOutputWithPast = model(**candidate, output_hidden_states=True)
          
          # TODO add back in the skip logits computation optimization
          # ensure logits computation was skipped for memory / speed
          # saves memory, requires changing a line the transformers lib implementation of qwen
          # (or other LLM in used)
          #assert(query_outputs.logits is None)
          #assert(candidate_outputs.logits is None)

          q_eos_token_emb = get_last_token_embed(query["input_ids"], query_outputs.hidden_states[-1], 0)
          c_eos_token_emb= get_last_token_embed(candidate["input_ids"], candidate_outputs.hidden_states[-1], 0)

          loss, _ = self.gathered_loss(q_eos_token_emb,c_eos_token_emb) if # \
          else self.local_loss(q_eos_token_emb,c_eos_token_emb)

          return loss

     def compute_loss(self, model, inputs, return_outputs=False):
          if self.args.loss_type == "last_token":
               return self.last_token_loss(model, inputs, return_outputs=return_outputs)
          else:
               raise Exception("Loss type not implemented")
          
     def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
          return RandomSampler(self.train_dataset)
     
     def gathered_loss(self, q_emb, c_emb):
          """
          Compute the loss by gathering across GPUs.
          """
           # Get the number of GPUs (world_size)
          world_size = dist.get_world_size()
          rank = dist.get_rank()

          # Gather q_embed and c_embed from all GPUs
          q_global = [torch.zeros_like(q_emb) for _ in range(world_size)]
          c_global = [torch.zeros_like(c_emb) for _ in range(world_size)]

          dist.all_gather(q_global, q_emb)
          dist.all_gather(c_global, c_emb)
          
          q_global[rank] = q_emb
          c_global[rank] = c_emb

          # Concatenate the gathered embeddings along the batch dimension
          q_global = torch.cat(q_global, dim=0)
          c_global = torch.cat(c_global, dim=0)

          loss_global, acc_global = compute_contrastive_loss(q_global, c_global)
          loss_local, acc_local = compute_contrastive_loss(q_emb.detach(), c_emb.detach())

          self.log_to_wandb("global_accuracy", acc_global.detach())
          self.log_to_wandb("global_loss", loss_global.detach())
          self.log_to_wandb("local_accuracy", acc_local)
          self.log_to_wandb("local_loss", loss_local)
          return loss_global, acc_global
     
     def local_loss(self, q_emb, c_emb):
          """
          Compute the loss locally on each GPU, average later.
          """

          local_loss, local_acc = compute_contrastive_loss(q_emb, c_emb)

          self.log_to_wandb("global_accuracy", local_acc.detach())
          self.log_to_wandb("global_loss", local_acc.detach())

          return local_loss, local_acc

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

    accuracy = (max_idxs == sim_targets).sum() / bs

    return loss, accuracy

def get_last_token_embed(input_ids, hidden_state, padding_token_id):
    # Find the position of the last non-padding token for each sequence
    mask = input_ids != padding_token_id  # Create a mask where padding tokens are False
    last_token_pos = mask.sum(dim=1) - 1  # Get the index of the last non-padding token

    # Create a range tensor for batch indexing
    batch_size = input_ids.size(0)
    batch_range = torch.arange(batch_size, device=input_ids.device)

    # Extract the last token embedding for each sequence
    last_token_embeds = hidden_state[batch_range, last_token_pos]

    return last_token_embeds