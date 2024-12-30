import torch
from monkey_patch.qwen_attn_patch import monkey_patch_transformers_lib, unmask_attn_monkey_patch
from qwen.vision_process import process_vision_info
import functools
from peft import LoraConfig, get_peft_model

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
        from transformers import AutoProcessor, AutoModel
        from internvl.model.abc.modeling_abc import abcQwenVL
     
     
        # Load base model
        model = abcQwenVL.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )

        target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        
        lora_config = LoraConfig(
            r=64,
            target_modules=target_modules,
            lora_alpha=2*64,
            lora_dropout=0.05,
            #task_type='CAUSAL_LM', # Dictates params are passed to the underlying HG model by the PEFT wrapper.
            use_dora=False
        )

        model.model = get_peft_model(model.model, lora_config)

        target_modules = ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']
        lora_config = LoraConfig(
            r=64,
            target_modules=target_modules,
            lora_alpha=2*64,
            lora_dropout=0.05,
            use_dora=False
        )
        model.visual = get_peft_model(model.visual, lora_config)
        print("applied LoRA")
#        l = list((model.state_dict().keys()))
#        torch.save(l, "/home/b3schnei/model.keys")

        # Load base model LoRA local model
        model = model.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

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

            return model.embed(_prepare_input(inps)).cpu()
        return functools.partial(embed, model, processor)