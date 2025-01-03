import torch
from monkey_patch.qwen_attn_patch import monkey_patch_transformers_lib, unmask_attn_monkey_patch
from qwen.vision_process import process_vision_info
import functools
from peft import LoraConfig, get_peft_model, PeftModel
from collections.abc import Mapping

def _prepare_input(data):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        cuda_tensor = data.cuda()
        return cuda_tensor
    return data

def get_model_with_embed_function(model_type, model_path):
    if model_type == "abcQwenVL":
        monkey_patch_transformers_lib()
        unmask_attn_monkey_patch()
        min_pixels = 256*28*28
        max_pixels = 512*28*28
        from transformers import AutoProcessor
        from internvl.model.abc.modeling_abc import abcQwenVL
     
     
        # Load base model
        base_model = abcQwenVL.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(base_model, "/home/b3schnei/output/QwenVL-8B-Large-Batch")

        model = model.merge_and_unload()
        model.to(torch.bfloat16).cuda()

        processor = AutoProcessor.from_pretrained(model_path,
                                                    padding_side="right",
                                                    use_fast=False,
                                                    max_pixels=max_pixels,
                                                    min_pixels=min_pixels)
        
    
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

            inps = _prepare_input(inps)
            output = model.embed(inps)
            return output.cpu()
        return functools.partial(embed, model, processor)