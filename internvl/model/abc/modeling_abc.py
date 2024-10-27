from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .abc_util import *
from torch import nn
import math 
from torch.nn import init

class IVLTCO(InternVLChatModel):
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    """
    
    attn_mask = "bidirectional"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.temperature = nn.Parameter(torch.tensor(0.1,
                                                            requires_grad=True,
                                                            dtype=torch.float32))

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

        loss, acc = compute_contrastive_loss(q_emb, c_emb, temperature=self.temperature.float())
        
        outputs = {}
        if return_outputs:
            temperature = self.temperature.data.detach().cpu()
            # zero3 guard
            if temperature.numel() == 0: temperature = torch.tensor([-1])

            outputs["accuracy"] = acc
            outputs["temperature"] = temperature
        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }

        return (loss, outputs) if return_outputs or return_prediction else loss

class IVLTCOS(InternVLChatModel):
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    and label smoothing.
    """
    
    attn_mask = "bidirectional"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.temperature = nn.Parameter(torch.tensor(0.1,
                                                            requires_grad=True,
                                                            dtype=torch.float32))

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

        loss, acc = compute_contrastive_loss(q_emb, c_emb, temperature=self.temperature.float(), label_smoothing=0.1)
        
        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
            outputs["temperature"] = self.temperature
        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }
        return (loss, outputs) if return_outputs or return_prediction else loss

class IVLTCOS(InternVLChatModel):
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    and label smoothing.
    """
    
    attn_mask = "bidirectional"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.temperature = nn.Parameter(torch.tensor(0.1,
                                                            requires_grad=True,
                                                            dtype=torch.float32))

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

        loss, acc = compute_contrastive_loss(q_emb, c_emb, temperature=self.temperature.float(), label_smoothing=0.1)
        
        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
            outputs["temperature"] = self.temperature
        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }
        return (loss, outputs) if return_outputs or return_prediction else loss

# Use custom iniatialization for linear layers to prevent very large values for bias.
class Linear(nn.Linear):

    def reset_parameters(self) -> None:
        init.eye_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

class MLP(nn.Module):

    def __init__(self, embed_size: int, hidden_size: int=None):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size if hidden_size is not None else embed_size

        # Initialize in high precision to prevent xavier init from underflowing
        self.linear_layer1 = Linear(self.embed_size, self.hidden_size, dtype=torch.float32)
        self.linear_layer2 = Linear(self.hidden_size, self.embed_size, dtype=torch.float32)
        self.act = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.linear_layer1(x)
        y = self.act(y)
        y = self.linear_layer2(y)
        out = x+y
        return out

class IVLMLP(InternVLChatModel):
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    and label smoothing.
    and MLP projection layer.
    """
    
    attn_mask = "bidirectional"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mlp_head = MLP(config.llm_config.hidden_size)
        self.temperature = nn.Parameter(torch.tensor(0.1,
                                                            requires_grad=True,
                                                            dtype=torch.float32))
        
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
        
        q_emb = self.mlp_head(q_eos_token_emb).float()
        c_emb = self.mlp_head(c_eos_token_emb).float()
        q_emb = F.normalize(q_emb, dim=-1)
        c_emb = F.normalize(c_emb, dim=-1)

        loss, acc = compute_contrastive_loss(q_emb, c_emb, temperature=self.temperature.float(), label_smoothing=0.1)
        
        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
            outputs["temperature"] = self.temperature
        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }
        return (loss, outputs) if (return_outputs or return_prediction) else loss

class IVLMLPL(InternVLChatModel):
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    and label smoothing.
    and MLP projection layer.
    """
    
    attn_mask = "bidirectional"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mlp_head = MLP(config.llm_config.hidden_size, hidden_size=4096)
        self.temperature = nn.Parameter(torch.tensor(0.1,
                                                            requires_grad=True,
                                                            dtype=torch.float32))
        
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
        
        q_emb = self.mlp_head(q_eos_token_emb).float()
        c_emb = self.mlp_head(c_eos_token_emb).float()
        q_emb = F.normalize(q_emb, dim=-1)
        c_emb = F.normalize(c_emb, dim=-1)

        loss, acc = compute_contrastive_loss(q_emb, c_emb, temperature=self.temperature.float(), label_smoothing=0.1)
        
        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
            outputs["temperature"] = self.temperature
        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }
        return (loss, outputs) if (return_outputs or return_prediction) else loss

class IVLMLPLG(InternVLChatModel):
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    and label smoothing.
    and MLP projection layer.
    """
    
    attn_mask = "bidirectional"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mlp_head = MLP(config.llm_config.hidden_size, hidden_size=4096)
        self.temperature = nn.Parameter(torch.tensor(0.07, requires_grad=True, dtype=torch.float32))
        
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
        
        q_emb = self.mlp_head(q_eos_token_emb).float()
        c_emb = self.mlp_head(c_eos_token_emb).float()
        q_emb = F.normalize(q_emb, dim=-1)
        c_emb = F.normalize(c_emb, dim=-1)
        
        # eval batch_size is num_gpus * eval_per_gpu
        loss, acc, bs = compute_gathered_loss(q_emb, c_emb, temperature=self.temperature.float(), label_smoothing=0.1)

        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
            outputs["temperature"] = self.temperature.data.clone()
            outputs["batch_size"] = bs

        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }
        return (loss, outputs) if (return_outputs or return_prediction) else loss


class IVLMLP2(InternVLChatModel):
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    and label smoothing.
    and MLP projection layer.
    """
    
    attn_mask = "bidirectional"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mlp_q = MLP(config.llm_config.hidden_size)
        self.mlp_c = MLP(config.llm_config.hidden_size)
        self.temperature = nn.Parameter(torch.tensor(0.1,
                                                            requires_grad=True,
                                                            dtype=torch.float32))
        
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
        
        q_emb = self.mlp_q(q_eos_token_emb).float()
        c_emb = self.mlp_c(c_eos_token_emb).float()
        q_emb = F.normalize(q_emb, dim=-1)
        c_emb = F.normalize(c_emb, dim=-1)

        loss, acc = compute_contrastive_loss(q_emb, c_emb, temperature=self.temperature.float(), label_smoothing=0.1)
        
        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
            outputs["temperature"] = self.temperature
        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }
        return (loss, outputs) if (return_outputs or return_prediction) else loss




# TODO add update these model archs to support
# 1. Gathered loss
# 2. Cross attn adpater head
# 3. MLP Adapter head
# 4. Hard negatives
MODEL_ARCHITECTURE = {
    "IVLTCO": IVLTCO,
    "IVLTCOS": IVLTCOS,
    "IVLMLP": IVLMLP,
    "IVLMLP2": IVLMLP2,
    "IVLMLPL": IVLMLPL,
    "IVLMLPLG": IVLMLPLG
 }