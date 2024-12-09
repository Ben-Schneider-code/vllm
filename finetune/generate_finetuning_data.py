# ----------------
# Dervied from: 
# https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language.py#L447
# ----------------


import os
os.environ["WANDB_PROJECT"] = "WIKIWEB_INFERENCE"
import sys
import torch
import wandb
from vllm import LLM, SamplingParams
from finetune.dataset import InstructionFiltering, qwen_collator
import orjson
import math
import time

# CONFIG ---------------
PROMPT = "Using the image and corresponding desciption write 3 questions about the image that are answered in the descrption. Fill in the following json template with both the questions and their corresponding answer from the description: {Question1: <your question 1 here>, Answer1: <your answer 1 here>, Question2: <your question 2 here>, Answer2: <your answer 2 here>, Question3: <your question 3 here>, Answer3: <your answer 3 here>}"
MAX_TOKENS = 21_000
BATCH_SIZE = 32
TEMPERATURE = 0.0  # For deterministic output, set temperature to 0.
TOP_P = 1.0
BATCH_SIZE = 16
min_item, max_item = int(sys.argv[1]), int(sys.argv[2])
model_name = "Qwen/Qwen2-VL-72B-Instruct"
modality = "image"
full_dataset= InstructionFiltering(PROMPT)
output_dir = sys.argv[3]
run_name = f"instruct_wikiweb_{min_item}_to_{max_item}_of_{len(full_dataset)}"
# -----------------------

log_config = {
    "NUM_ITEMS": max_item-min_item,
    "MIN_IDX": min_item,
    "MAX_IDX": max_item,
    "NUM_BATCHES": math.ceil((max_item-min_item)/BATCH_SIZE)
}

wandb.init(name=run_name, config=log_config)


llm = LLM(
    model=model_name,
    max_model_len=21327, # max kv cache for our model
    tensor_parallel_size=torch.cuda.device_count(),
    max_num_seqs=5,
    # Note - mm_processor_kwargs can also be passed to generate/chat calls
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    },
)

# Define your sampling parameters
sampling_params = SamplingParams(
    n=1,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    stop=[]
)

from torch.utils.data import DataLoader, Subset
# Subset of indices that are run on this node
dataset = Subset(full_dataset, range(min_item, max_item))
dl = DataLoader(dataset, num_workers=8, collate_fn=qwen_collator, batch_size=BATCH_SIZE, shuffle=False, prefetch_factor=4)

save_dict = {}

run_start_time = time.time()
begin_of_batch = None
end_of_batch = 0
yield_time = None

for batch_num, (idx_list, batch) in enumerate(dl):

    begin_of_batch = time.time()
    yield_time = begin_of_batch - end_of_batch

    try:        
        outputs = llm.generate(batch, sampling_params=sampling_params)
        for idx, out in zip(idx_list,outputs):
            save_dict[str(idx)] = out.outputs[0].text    
    except Exception as e:
        print(e)

    end_of_batch = time.time()
    
    wandb.log({"CURRENT_BATCH": batch_num,
                "SECONDS_COMPUTE_BATCH": end_of_batch-begin_of_batch,
                "DATALOADER_YIELD_OVERHEAD": yield_time,
                "TOTAL_BATCH_TIME": yield_time+(end_of_batch-begin_of_batch),
                "TOTAL_RUNTIME": time.time() - run_start_time,
                "SECONDS_PER_ITEM": (time.time() - run_start_time) / (batch_num*BATCH_SIZE)
                })

filename = f"{run_name}.json"
save_data_path = os.path.join(output_dir, filename)
os.makedirs(os.path.dirname(save_data_path), exist_ok=True)
with open(save_data_path, "wb") as output_file:
    output_file.write(orjson.dumps(save_dict, option=orjson.OPT_INDENT_2))

artifact = wandb.Artifact(filename, type="preprocessed-data")
artifact.add_file(save_data_path)
wandb.log_artifact(artifact)
wandb.finish()