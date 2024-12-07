# ----------------
# Dervied from: 
# https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language.py#L447
# ----------------


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from vllm import LLM, SamplingParams
from finetune.dataset import InstructionFiltering

MAX_TOKENS = 4096
TEMPERATURE = 0.0  # For deterministic output, set temperature to 0.
TOP_P = 1.0
BATCH_SIZE = 1

# Load the dataset
dataset = InstructionFiltering("")
image, text = dataset[100]
image = image.convert("RGB")

model_name = "Qwen/Qwen2-VL-72B-Instruct"
modality = "image"

# Qwen2-VL
def run_qwen2_vl(question: str, modality: str):
    assert modality == "image"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        tensor_parallel_size=4,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    )

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids

llm, prompt, stop_token_ids = run_qwen2_vl("what is in this image?", modality)

# Define your sampling parameters
sampling_params = SamplingParams(
    n=1,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    stop=[]
)


# Batch inference
inputs = [{
    "prompt": prompt,
    "multi_modal_data": {
        modality: image
    },
} for _ in range(128)]

outputs = llm.generate(inputs, sampling_params=sampling_params)

for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)