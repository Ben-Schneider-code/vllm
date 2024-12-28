import torch
from monkey_patch.qwen_attn_patch import monkey_patch_transformers_lib, unmask_attn_monkey_patch
from qwen.vision_process import process_vision_info
import functools

def _prepare_input(data):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    return data

def get_model_with_embed_function(model_type, model_path):
    if model_type == "abcQwenVL":
        monkey_patch_transformers_lib()
        unmask_attn_monkey_patch()
        min_pixels = 256*28*28
        max_pixels = 512*28*28
        from transformers import AutoProcessor
        from internvl.model.abc.modeling_abc import abcQwenVL

        processor = AutoProcessor.from_pretrained(model_path,
                                                    padding_side="right",
                                                    use_fast=False,
                                                    max_pixels=max_pixels,
                                                    min_pixels=min_pixels)

        model = abcQwenVL.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).cuda()

        def embed(model, processor, item: str = "", dtype: str = "text"):
            assert dtype in ["image", "text"]

            conversation = None

            if dtype == "text":
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text" : item}
                    ]
                }]
            else:
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item},
                        {"type": "text", "text" : ""}
                    ]
                }]
            text_input = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(conversation)

            inps = processor(
                text=text_input,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            return model.embed(_prepare_input(inps)).cpu()
        return functools.partial(embed, model, processor)