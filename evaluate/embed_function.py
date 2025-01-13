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

def get_abcQwenVL(model_type, model_path):
        monkey_patch_transformers_lib()
        unmask_attn_monkey_patch()
        min_pixels = 128*28*28
        max_pixels = 1024*28*28
        from transformers import AutoProcessor
        from internvl.model.abc.modeling_abc import abcQwenVL
     
     
        # Load base model
        base_model = abcQwenVL.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(base_model, model_path)

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

def get_abcQwenVL_instruct(model_type, model_path, instruct_model):
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
        
        # Load and merge pretrain adapter
        pretrained_model = PeftModel.from_pretrained(base_model, model_path)
        pretrained_model = pretrained_model.merge_and_unload()

        # Load instruction model
        pretrained_model.instruction_mode = True
        model = PeftModel.from_pretrained(pretrained_model, instruct_model)
        
        # The forward method needs to be able to toggle LoR
        setattr(model.get_base_model(), "get_peft_wrapper", lambda: model)
        model.to(torch.bfloat16).cuda()

        processor = AutoProcessor.from_pretrained(model_path,
                                                    padding_side="right",
                                                    use_fast=False,
                                                    max_pixels=max_pixels,
                                                    min_pixels=min_pixels)
        
    
        def embed(model, processor, item: str = "", dtype: str = "text", instruction=""):
            assert dtype in ["image", "text"]

            conversation = None

            if dtype == "text":
                
                conv_templates = [[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text" : i}
                    ]
                }] for i in item]

                text_input =[ processor.apply_chat_template(
                    i, tokenize=False, add_generation_prompt=True
                ) for i in conv_templates]

                image_inputs = None

            else:
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item},
                        {"type": "text", "text" : f"Instruction: {instruction}"}
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
        print("\nLoad instruction version of model\n")
        return functools.partial(embed, model, processor)   

def get_model_with_embed_function(model_type, pretrain_model_path, instruct_model_path=None):
    if model_type == "abcQwenVL":
        return get_abcQwenVL(model_type, pretrain_model_path)
    elif model_type == "abcQwenVL-Instruct":
        return get_abcQwenVL_instruct(model_type, pretrain_model_path, instruct_model_path)
    else:
        raise Exception("NotImplementedError")
