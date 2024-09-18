
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.distributed as dist
from .abc_util import *

class abc_mean_token(InternVLChatModel):

    def forward(self, inputs, return_outputs=False):
        query = inputs["query"]
        candidate = inputs["pos_cand"]
        query.pop("labels")
        candidate.pop("labels")

        query_outputs : CausalLMOutputWithPast = super(**query, output_hidden_states=True)
        candidate_outputs : CausalLMOutputWithPast = super(**candidate, output_hidden_states=True)
        
        # ensure logits computation was skipped for memory / speed
        # (or other LLM in used)
        assert(query_outputs.logits is None)
        assert(candidate_outputs.logits is None)

        q_eos_token_emb = get_mean_token_embed(query["input_ids"], query_outputs.hidden_states[-1], 0)
        c_eos_token_emb= get_mean_token_embed(candidate["input_ids"], candidate_outputs.hidden_states[-1], 0)
        loss, acc = self.compute_loss(q_eos_token_emb,c_eos_token_emb,return_outputs)

        return (loss, {"accuracy": acc}) if return_outputs else loss

    def compute_loss(self, q_emb, c_emb, return_outputs):
        """
        Compute the loss locally on each GPU, average later.
        """
        q_emb = q_emb.float()
        c_emb = c_emb.float()

        local_loss, local_acc = compute_contrastive_loss(q_emb, c_emb)

        # reduce for logging
        log_acc = local_acc.detach().clone()
        log_loss = local_loss.detach().clone()
        world_size = dist.get_world_size()
        dist.all_reduce(log_acc.detach(),op=dist.ReduceOp.SUM)
        dist.all_reduce(log_loss.detach(),op=dist.ReduceOp.SUM)
        
        # log only on main process
        if not return_outputs and dist.get_rank() == 0:
            self.log_to_wandb("local_accuracy", log_acc/world_size)
            self.log_to_wandb("local_loss", log_loss/world_size)

        return local_loss, local_acc

class abc_last_token(abc_mean_token):
    
     def forward(self, model, inputs, return_outputs=False):

          query = inputs["query"]
          candidate = inputs["pos_cand"]

          query.pop("labels")
          candidate.pop("labels")
          query_outputs : CausalLMOutputWithPast = super(**query, output_hidden_states=True)
          candidate_outputs : CausalLMOutputWithPast = model(**candidate, output_hidden_states=True)
          
          assert(query_outputs.logits is None)
          assert(candidate_outputs.logits is None)

          q_eos_token_emb = get_last_token_embed(query["input_ids"], query_outputs.hidden_states[-1], 0)
          c_eos_token_emb= get_last_token_embed(candidate["input_ids"], candidate_outputs.hidden_states[-1], 0)

          loss, acc = self.compute_loss(q_eos_token_emb,c_eos_token_emb,return_outputs)

          return (loss, {"accuracy": acc}) if return_outputs else loss