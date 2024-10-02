import torch
from typing import Optional, List, Union, Tuple
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
import transformers.models
import transformers.models.qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention,Qwen2FlashAttention2,Qwen2SdpaAttention
from internvl.model.internlm2.modeling_internlm2 import InternLM2FlashAttention2

class ModifiedQwen2Attention(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2FlashAttention2(Qwen2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2SdpaAttention(Qwen2SdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


QWEN2_ATTENTION_UNMASKED = {
    "eager": ModifiedQwen2Attention,
    "flash_attention_2": ModifiedQwen2FlashAttention2,
    "sdpa": ModifiedQwen2SdpaAttention,
}

class ModifiedInternLMFlashAttention2(InternLM2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

INTERNLM2_ATTENTION_UNMASKED = {
    "eager": None,
    "flash_attention_2": ModifiedInternLMFlashAttention2,
}

def internlm2_forward_low_memory(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = None
    loss = None

    if not return_dict:
        raise Exception("Not supported")

    output = CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    
    return output

def qwen2_forward_low_memory(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = None
        loss = None

        if not return_dict:
            raise Exception("Not supported")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def unmask_attn_monkey_patch():
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES = QWEN2_ATTENTION_UNMASKED
    import internvl.model.internlm2.modeling_internlm2 as internlm2
    internlm2.INTERNLM2_ATTENTION_CLASSES = INTERNLM2_ATTENTION_UNMASKED

def forward_memory_opt_monkey_patch():
    """
    Skip the logits computation.
    The logits computation [bs, seq_len, vocab] is a massive tensor that gets upcast to fp32. 
    However we only use hidden state embeds, so we don't need it. :)
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    Qwen2ForCausalLM.forward = qwen2_forward_low_memory
    from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
    InternLM2ForCausalLM.forward = internlm2_forward_low_memory