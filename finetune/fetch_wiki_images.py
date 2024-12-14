import os
import requests
from PIL import Image
from io import BytesIO
import orjson
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

file_path = os.environ["WIKI_INSTRUCT_ROOT"]
image_fldr = os.path.join(os.path.dirname(file_path), "images")
Path(image_fldr).mkdir(parents=True, exist_ok=True)

def fetch_item(id, image_url):
    try:
        # Define a User-Agent header for Wikimedia requests
        headers = {
            "User-Agent": "University_of_Waterloo_DataMining/1.0 contact:benjamin.schneider@uwaterloo.ca)"
        }

        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()  # Check if the request was successful
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        save_path = os.path.join(image_fldr, str(id) + ".jpeg")
        image.save(save_path)
    except Exception as e:
        print(f"Error fetching image {id} from {image_url}: {e}")

# Load metadata
with open(file_path, 'rb') as f:
    meta = orjson.loads(f.read())

# Prepare for parallel execution
def process_item(item):
    fetch_item(item["id"], item["url"])

# Use ThreadPoolExecutor for parallelization
with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust max_workers as needed
    futures = {executor.submit(process_item, item): item for item in meta}

    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            future.result()  # Retrieve result or handle exception
        except Exception as e:
            item = futures[future]
            print(f"Failed to process item {item['id']}: {e}")