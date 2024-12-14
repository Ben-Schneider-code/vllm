from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .abc_util import *
from torch import nn
from torch.nn import init

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

class IVLMLPLG(InternVLChatModel):
    
    """
    Added scaling to the contrastive loss and optimize the scaling param.
    and label smoothing.
    and MLP projection layer.
    """
    
    supports_gradient_checkpointing = True
    attn_mask = "bidirectional"
    instruction_mode = False

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
        loss, acc, num_cand = compute_gathered_loss(q_emb, c_emb, temperature=self.temperature.float(), label_smoothing=0.1)

        outputs = {}
        if return_outputs:
            outputs["accuracy"] = acc
            outputs["temperature"] = self.temperature.data.clone()
            outputs["num_cand"] = num_cand

        if return_prediction:
            outputs["prediction"] = {
                "meta": inputs["meta"],
                "q": q_emb.detach().cpu(),
                "c": c_emb.detach().cpu()
            }
        return (loss, outputs) if (return_outputs or return_prediction) else loss

# TODO add update these model archs to support
# 1. Hard negatives
MODEL_ARCHITECTURE = {
    "IVLMLPLG": IVLMLPLG
 }