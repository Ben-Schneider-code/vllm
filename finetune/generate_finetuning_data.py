from vllm import LLM, SamplingParams
import orjson
import os


data_path = os.environ["WIKIWEB"]

with open(data_path, "rb") as file:
    data = orjson.loads(file.read())

item = data[0]