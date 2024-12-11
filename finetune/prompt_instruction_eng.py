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
#PROMPT = "What are 3 interesting thngs about this image that are answered in the description? Fill in the following json template with both the interesting things and their corresponding answer from the description: {Thing 1: <your thing 1 here>, Answer1: <your answer 1 here>, Thing 2: <your thing 2 here>>, Answer2: <your answer 2 here>, Thing 1: <your thing 1 here>, Answer3: <your answer 3 here>}"
PROMPT = 'Give me 8 prompts a user might ask about this image and the corresponding answers. The prompts should require interpreting the image to answer. Each prompt and its corresponding answer should not have words in common. The prompts and answers should be full sentences. Fill in the following json template with both the prompts and their corresponding answers: {"Prompt 1": <your prompt 1 here>, Answer1: <your answer 1 here>, "Prompt 2": <your prompt 2 here>, Answer2: <your answer 2 here>, "Prompt 3": <your prompt 3 here>, Answer3: <your answer 3 here>, "Prompt 4": <your promp 4 here>, Answer4: <your answer 4 here>, "Prompt 5": <your prompt 5 here>, Answer5: <your answer 5 here>, "Prompt 6": <your query 6 here>, Answer6: <your answer 6 here>, "Prompt 7": <your query 7 here>, Answer7: <your answer 7 here>, "Prompt 8": <your question 8 here>, Answer8: <your answer 8 here>}'


MAX_TOKENS = 20_000
TEMPERATURE = 0.2  # We want some temperature for lexical diversity
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
dl = DataLoader(dataset, num_workers=2, collate_fn=qwen_collator, batch_size=BATCH_SIZE, shuffle=False)

save_dict = {}

run_start_time = time.time()
begin_of_batch = None
end_of_batch = 0
yield_time = None

for batch_num, (idx_list, meta, batch) in enumerate(dl):

    begin_of_batch = time.time()
    yield_time = begin_of_batch - end_of_batch

    #try:        
    outputs = llm.generate(batch, sampling_params=sampling_params)
    for idx, meta, out in zip(idx_list, meta, outputs):
        with open("output_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write("----------- Article ------------\n")
            log_file.write(f"{meta['article_url']}\n")
            log_file.write("----------- Image -----------\n")
            log_file.write(f"{meta['image_url']}\n")
            log_file.write("------- Model Response ------\n")
            model_text = out.outputs[0].text
            log_file.write(model_text)
            log_file.write("\n")  # Add a newline for separation      
            
    #except Exception as e:
    #    print(e)

    end_of_batch = time.time()
    
    wandb.log({"CURRENT_BATCH": batch_num,
                "SECONDS_COMPUTE_BATCH": end_of_batch-begin_of_batch,
                "DATALOADER_YIELD_OVERHEAD": yield_time,
                "TOTAL_BATCH_TIME": yield_time+(end_of_batch-begin_of_batch),
                "TOTAL_RUNTIME": time.time() - run_start_time,
                "SECONDS_PER_ITEM": (time.time() - run_start_time) / ((1+batch_num)*BATCH_SIZE)
                })


wandb.finish()