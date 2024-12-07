import os
import orjson
import numpy as np
data_path = os.environ["WIKIWEB"]

# Load the dataset
with open(data_path, "rb") as file:
    data = orjson.loads(file.read())

# Filter for files wiki pages with images
def filter_for_images(data):
    data = [i if sum(i['section_contains_images']) > 0 else None for i in data]
    return list(filter(lambda item: item is not None, data))

def article_length_arr(data):
    def sum_string_list(l):
        return sum([len(i) for i in l])
    return np.array([sum_string_list(i["section_text"]) for i in data])


data = filter_for_images(data)
lens = article_length_arr(data)


# Function to filter data for entries where lens is at least 5000
def filter_by_length(data, lens, threshold=1):
    return [entry for entry, length in zip(data, lens) if length >= threshold]

# Apply the function
data = filter_by_length(data, lens, threshold=8000)

# Save the filtered data to a file
filtered_data_path = "filtered_wikiweb.json"

with open(filtered_data_path, "wb") as output_file:
    output_file.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

print(f"Filtered data saved to {filtered_data_path}")

