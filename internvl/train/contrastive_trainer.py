from typing import Optional
import torch
from transformers import Trainer
from transformers.trainer import RandomSampler
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.distributed as dist

class ContrastiveTrainer(Trainer):

     # still need to add loss computation
     def compute_loss(self, model, inputs, return_outputs=False):
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
          
          # ensure logits computation was skipped for memory / speed
          # saves memory, requires changing a line the transformers lib implementation of qwen
          # (or other LLM in used)
          assert(query_outputs.logits is None)
          assert(candidate_outputs.logits is None)

          batch_idx = torch.arange(query["input_ids"].shape[0])
           
          q_embed = query_outputs.hidden_states[-1][batch_idx,query_token_idx].float()
          c_embed = candidate_outputs.hidden_states[-1][batch_idx,cand_token_idx].float()

          # Get the number of GPUs (world_size)
          world_size = dist.get_world_size()
          rank = dist.get_rank()

          # Gather q_embed and c_embed from all GPUs
          q_embed_list = [torch.zeros_like(q_embed) for _ in range(world_size)]
          c_embed_list = [torch.zeros_like(c_embed) for _ in range(world_size)]

          dist.all_gather(q_embed_list, q_embed)
          dist.all_gather(c_embed_list, c_embed)
          
          q_embed_list[rank] = q_embed
          c_embed_list[rank] = c_embed

          # Concatenate the gathered embeddings along the batch dimension
          q_embed_gathered = torch.cat(q_embed_list, dim=0)
          c_embed_gathered = torch.cat(c_embed_list, dim=0)

          loss, acc = compute_contrastive_loss(q_embed_gathered,c_embed_gathered)

          # torch.cuda.memory._dump_snapshot("/home/b3schnei/memory_snap_8.pickle")
          return loss

     def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
          return RandomSampler(self.train_dataset)
     
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