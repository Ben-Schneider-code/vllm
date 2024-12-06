import os
import orjson

data_path = os.environ["WIKIWEB"]

# Load the dataset
with open(data_path, "rb") as file:
    data = orjson.loads(file.read())

