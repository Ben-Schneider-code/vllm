import torch
from monkey_patch.qwen_attn_patch import monkey_patch_transformers_lib, unmask_attn_monkey_patch
from qwen.vision_process import process_vision_info
import functools
from peft import PeftModel
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
        min_pixels = 256*28*28
        max_pixels = 1024*28*28
        from transformers import AutoProcessor
        from model.modeling_abc import abcQwenVL
     
     
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

def get_abcQwenVL_instruct_model(model_type, model_path, instruct_model):
        from model.modeling_abc import abcQwenVL
        base_model = abcQwenVL.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
        
        base_model.instruction_mode = True

        # Load and merge pretrain adapter
        pretrained_model = PeftModel.from_pretrained(base_model, model_path)
        pretrained_model = pretrained_model.merge_and_unload()

        # Load instruction model
        model = PeftModel.from_pretrained(pretrained_model, instruct_model, adapter_name="instruct")
        
        # The forward method needs to be able to toggle LoRA
        setattr(model.get_base_model(), "get_peft_wrapper", lambda: model)
        model.to(torch.bfloat16).cuda()
        return model

def get_abcQwenVL_instruct_batch(model_type, model_path, instruct_model):
        monkey_patch_transformers_lib()
        unmask_attn_monkey_patch()
        min_pixels = 256*28*28
        max_pixels = 512*28*28
        from transformers import AutoProcessor
        
        model = get_abcQwenVL_instruct_model(model_type, model_path, instruct_model)

        processor = AutoProcessor.from_pretrained(model_path,
                                                    padding_side="right",
                                                    use_fast=False,
                                                    max_pixels=max_pixels,
                                                    min_pixels=min_pixels)
        
    
        def embed(model, processor, item: str = "", dtype: str = "text", instruction=""):
            assert dtype in ["image", "text"]

            conversation = None

            if dtype == "text":
                assert isinstance(item, list)
                text_inputs = []
                for i in item:
                    conversation = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text" : i}
                        ]
                    }]
                    text_input = processor.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                    text_inputs.append(text_input)

                inps = processor(
                    text=text_inputs,
                    images=None,
                    padding=True,
                    return_tensors="pt",
                )
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

            output = model.inst_embed(inps, dtype=="text")
            return output.cpu()
        print("\nLoad instruction version of model\n")
        return functools.partial(embed, model, processor)   

def get_abcQwenVL_instruct(model_type, model_path, instruct_model):
        monkey_patch_transformers_lib()
        unmask_attn_monkey_patch()
        min_pixels = 256*28*28
        max_pixels = 512*28*28
        from transformers import AutoProcessor
        
        model = get_abcQwenVL_instruct_model(model_type, model_path, instruct_model)

        processor = AutoProcessor.from_pretrained(model_path,
                                                    padding_side="right",
                                                    use_fast=False,
                                                    max_pixels=max_pixels,
                                                    min_pixels=min_pixels)
        
    
        def embed(model, processor, item: str = "", dtype: str = "text", instruction=""):
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

            output = model.inst_embed(inps, dtype=="text")
            return output.cpu()
        print("\nLoad instruction version of model\n")
        return functools.partial(embed, model, processor)   

def get_model_with_embed_function(model_type, pretrain_model_path, instruct_model_path=None, batch=False):
    if model_type == "abcQwenVL":
        return get_abcQwenVL(model_type, pretrain_model_path)
    elif model_type == "abcQwenVL-Instruct" and batch:
        return get_abcQwenVL_instruct_batch(model_type, pretrain_model_path, instruct_model_path)
    elif model_type == "abcQwenVL-Instruct":
        return get_abcQwenVL_instruct(model_type, pretrain_model_path, instruct_model_path)
    elif model_type == "vlm2vec":
        from eval_models.VLM2Vec.scripts.vlm2vec_functional import vlm2vec_embed_function
        return vlm2vec_embed_function()
    elif model_type == "magiclens":
        from eval_models.magiclens.magiclens_functional import magiclens_embed_function
        return magiclens_embed_function()
    elif model_type == "uniir":
        from eval_models.UniIR.src.models.uniir_clip.clip_scorefusion.uniir_functional import uniir_embed_function
        return uniir_embed_function()
    else:
        raise Exception("NotImplementedError")
