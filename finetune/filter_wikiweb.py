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

import matplotlib.pyplot as plt

# Plot the histogram
#plt.figure(figsize=(10, 6))
#lt.hist(lens, bins=30, edgecolor='black', alpha=0.7)
#plt.title('Histogram of Article Lengths')
#plt.xlabel('Article Length (characters)')
#plt.ylabel('Frequency')
#plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()