from dataset_utils.conceptual_captions import ConceptualCaptionsPretrainAdapter
import os  
import base64
from openai import AzureOpenAI  
import orjson
import sys

from qwen.qwen_dataset import get_split

api_key_path = sys.argv[1]
output_file_path = sys.argv[2]
ds = get_split(ConceptualCaptionsPretrainAdapter(negatives=None), pretrain=False)
root = ds.root

start, end  = 0, 2

with open(api_key_path, "rb") as file:  # Use "rb" mode for orjson
    data = orjson.loads(file.read())


endpoint = data["ENDPOINT"]
subscription_key = data["KEY"]
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  

client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",  
)

def query_gpt(ds, idx):

    item = ds[idx]

    IMAGE_PATH = os.path.join(root, item["query"]["image"])
    encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

    caption = item["pos_cand"]["conversations"][0]["value"]

    #Prepare the chat prompt 
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that helps people find information."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                },
                {
                    "type": "text",
                    "text": f"Caption: {caption}\nIdentify two distinct things in the image. Provide a corresponding user prompt for each thing in the image. Lastly, for each thing rewrite the caption to discuss only that thing. The caption and prompt should NOT have words in common." +
                    " Fill in the following json template with your answer: {\"prompt 1\": <prompt 1 here>, \"caption 1\": <caption 1 here>, \"prompt 2\": <prompt 2 here>, \"caption 2\": <caption 2 here>}"

                }
            ]
        },
    ] 
        
    # Include speech result if speech is enabled  
    messages = chat_prompt  

    # Generate the completion  
    completion = client.chat.completions.create(  
        model=deployment,  
        messages=messages,  
        max_tokens=800,  
        temperature=0.9,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,  
        stop=None,  
        stream=False
    )

    json_str = completion.to_json()
    comp_dict = orjson.loads(json_str)    

    return comp_dict, item

output = []
for i in range(start, end):

    try:
        completion, item = query_gpt(ds, i)
        o = {
            "id": item["id"],
            "item": item,
            "response": completion.to_json()

        }
        output.append(o)

    except:
        print(f"fail on index {i}")

# Save the output using orjson
with open(output_file_path, "wb") as file:  # Use "wb" mode for orjson
    file.write(orjson.dumps(output, option=orjson.OPT_INDENT_2))  # Pretty-print with indentation
