
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .abc_util import *

class IVLMT(InternVLChatModel):

    def forward(self, inputs, return_outputs=False, return_prediction=False):
        query = inputs["query"]
        candidate = inputs["pos_cand"]
        query.pop("labels")
        candidate.pop("labels")

        query_outputs : CausalLMOutputWithPast = super().forward(**query, output_hidden_states=True)
        candidate_outputs : CausalLMOutputWithPast = super().forward(**candidate, output_hidden_states=True)
        
        # ensure logits computation was skipped for memory / speed
        # (or other LLM in used)
        assert(query_outputs.logits is None)
        assert(candidate_outputs.logits is None)

        q_eos_token_emb = get_mean_token_embed(query["input_ids"], query_outputs.hidden_states[-1], 0)
        c_eos_token_emb= get_mean_token_embed(candidate["input_ids"], candidate_outputs.hidden_states[-1], 0)
        
        q_emb = q_eos_token_emb.float()
        c_emb = c_eos_token_emb.float()
        q_emb = F.normalize(q_emb, dim=-1)
        c_emb = F.normalize(c_emb, dim=-1)

        loss, acc = compute_contrastive_loss(q_emb, c_emb)
        
        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
        if return_prediction:
            outputs["prediction"] = {
                "q": q_emb.detach(),
                "c": c_emb.detach()
            }

        return (loss, outputs) if return_outputs else loss

class IVLLT(InternVLChatModel):
    
     def forward(self, inputs, return_outputs=False, return_prediction=False):

        query = inputs["query"]
        candidate = inputs["pos_cand"]

        query.pop("labels")
        candidate.pop("labels")
        query_outputs : CausalLMOutputWithPast = super().forward(**query, output_hidden_states=True)
        candidate_outputs : CausalLMOutputWithPast = super().forward(**candidate, output_hidden_states=True)
        
        assert(query_outputs.logits is None)
        assert(candidate_outputs.logits is None)

        q_eos_token_emb = get_last_token_embed(query["input_ids"], query_outputs.hidden_states[-1], 0)
        c_eos_token_emb= get_last_token_embed(candidate["input_ids"], candidate_outputs.hidden_states[-1], 0)

        q_emb = q_eos_token_emb.float()
        c_emb = c_eos_token_emb.float()
        q_emb = F.normalize(q_emb, dim=-1)
        c_emb = F.normalize(c_emb, dim=-1)
        loss, acc = compute_contrastive_loss(q_emb, c_emb)

        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
        if return_prediction:
            outputs["prediction"] = {
                "q": q_emb.detach(),
                "c": c_emb.detach()
            }

        return (loss, outputs) if return_outputs else loss