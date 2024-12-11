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
PROMPT = 'Give me 8 prompts a user might ask about this image and the corresponding answers. The prompts should require interpreting the image to answer. Each prompt and its corresponding answer should not have words in common. The prompts and answers should be full sentences. Fill in the following json template with both the prompts and their corresponding answers: {"Prompt 1": <your prompt 1 here>, Answer1: <your answer 1 here>, "Prompt 2": <your prompt 2 here>, Answer2: <your answer 2 here>, "Prompt 3": <your prompt 3 here>, Answer3: <your answer 3 here>, "Prompt 4": <your promp 4 here>, Answer4: <your answer 4 here>, "Prompt 5": <your prompt 5 here>, Answer5: <your answer 5 here>, "Prompt 6": <your query 6 here>, Answer6: <your answer 6 here>, "Prompt 7": <your query 7 here>, Answer7: <your answer 7 here>, "Prompt 8": <your question 8 here>, Answer8: <your answer 8 here>}'
MAX_TOKENS = 20_000
TEMPERATURE = 0.2  # We want some temperature for lexical diversity
TOP_P = 1.0
BATCH_SIZE = 16
min_item, max_item, num_items_in_range = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) 
model_name = "Qwen/Qwen2-VL-72B-Instruct"
modality = "image"
full_dataset= InstructionFiltering(PROMPT)
output_dir = sys.argv[4]
run_name = f"instruct_wikiweb_{min_item}_to_{max_item}_of_{len(full_dataset)}"
# -----------------------
total_batches = math.ceil(num_items_in_range/BATCH_SIZE)
log_config = {
    "PROMPT": PROMPT,
    "NUM_ITEMS": num_items_in_range,
    "MIN_IDX": min_item,
    "MAX_IDX": max_item,
    "NUM_BATCHES": total_batches
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
import random
sample_idx = random.sample(range(min_item, max_item), num_items_in_range)


dataset = Subset(full_dataset, sample_idx)
dl = DataLoader(dataset, num_workers=8, collate_fn=qwen_collator, batch_size=BATCH_SIZE, shuffle=False, prefetch_factor=4)

save_dict = {}

run_start_time = time.time()
begin_of_batch = None
end_of_batch = 0
yield_time = None

for batch_num, (idx_list,meta, batch) in enumerate(dl):

    begin_of_batch = time.time()
    yield_time = begin_of_batch - end_of_batch

    try:        
        outputs = llm.generate(batch, sampling_params=sampling_params)
        for idx, out in zip(idx_list,outputs):
            save_dict[str(idx)] = out.outputs[0].text    
    except Exception as e:
        print(e)

    end_of_batch = time.time()
    
    total_runtime = time.time() - run_start_time
    seconds_per_item = (time.time() - run_start_time) / ((1+batch_num)*BATCH_SIZE)

    wandb.log({"CURRENT_BATCH": batch_num,
                "SECONDS_COMPUTE_BATCH": end_of_batch-begin_of_batch,
                "DATALOADER_YIELD_OVERHEAD": yield_time,
                "TOTAL_BATCH_TIME": yield_time+(end_of_batch-begin_of_batch),
                "TOTAL_RUNTIME": total_runtime,
                "SECONDS_PER_ITEM": seconds_per_item,
                "TIME_REMAINING_IN_HOURS": (total_batches-batch_num-1) * BATCH_SIZE * seconds_per_item / 60**2 
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