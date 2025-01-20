import torch
from typing import Optional, List, Union, Tuple
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
import transformers.models.qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention,Qwen2FlashAttention2,Qwen2SdpaAttention
from internvl.model.internlm2.modeling_internlm2 import InternLM2FlashAttention2
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast

# Qwen LM ----------------------------------

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

# -----------------------------------------


# QwenVL ----------------------------------

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention, Qwen2VLFlashAttention2, Qwen2VLSdpaAttention

class ModifiedQwenVLAttention(Qwen2VLAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwenVLFlashAttention2(Qwen2VLFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwenVLSdpaAttention(Qwen2VLSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

QWENVL_ATTENTION_UNMASKED = {
    "eager": ModifiedQwenVLAttention,
    "flash_attention_2": ModifiedQwenVLFlashAttention2,
    "sdpa": ModifiedQwenVLSdpaAttention,
}

# -----------------------------------------

# QwenVL ----------------------------------

from transformers.models.mistral.modeling_mistral import MistralFlashAttention2

class ModifiedMistralFlashAttention2(MistralFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

MISTRAL_ATTENTION_UNMASKED = {
    "eager": lambda x: x,
    "flash_attention_2": ModifiedMistralFlashAttention2,
    "sdpa": lambda x: x,
}

# -----------------------------------------


# InternVL ----------------------------------

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

def qwenvl_low_memory_forward(
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
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
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

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

def unmask_attn_monkey_patch():
    
    transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES = MISTRAL_ATTENTION_UNMASKED
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES = QWEN2_ATTENTION_UNMASKED
    transformers.models.qwen2_vl.modeling_qwen2_vl.QWEN2_VL_ATTENTION_CLASSES = QWENVL_ATTENTION_UNMASKED
    import internvl.model.internlm2.modeling_internlm2 as internlm2
    internlm2.INTERNLM2_ATTENTION_CLASSES = INTERNLM2_ATTENTION_UNMASKED
    
def get_qwenvl_vision_tower_input_embeddings(self) -> torch.nn.Module:
    return self.patch_embed

def llava_low_mem_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    image_sizes: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_length=30)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )

    legacy_processing = False
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_features = None
    if pixel_values is not None and pixel_values.size(0) > 0:
        image_features = self.get_image_features(
            pixel_values,
            image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

        # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
        image_features, feature_lens = self.pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_newline=self.image_newline,
        )

    
    if image_features is not None:
        n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        num_logits_to_keep=num_logits_to_keep,
    )

    logits = None
    loss = None


    return LlavaNextCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )

def monkey_patch_transformers_lib():
    """
    Skip the logits computation.
    The logits computation [bs, seq_len, vocab] is a massive tensor that gets upcast to fp32. 
    However we only use hidden state embeds, so we don't need it. :)
    """

    from transformers.models.llava_next.modeling_llava_next import LlavaNextForConditionalGeneration
    LlavaNextForConditionalGeneration.forward = llava_low_mem_forward
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    Qwen2ForCausalLM.forward = qwen2_forward_low_memory
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    Qwen2VLForConditionalGeneration.forward = qwenvl_low_memory_forward
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
    Qwen2VisionTransformerPretrainedModel.get_input_embeddings = get_qwenvl_vision_tower_input_embeddings
    from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
    InternLM2ForCausalLM.forward = internlm2_forward_low_memory

    