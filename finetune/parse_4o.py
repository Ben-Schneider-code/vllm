import os
import orjson
import sys

# File path from the environment variable
file_path = sys.argv[1]
output_path = sys.argv[2]
keys = ["prompt 1", "prompt 2", "caption 1", "caption 2"]
required_keys_set = set(keys)

# Reading the file and loading its content
with open(file_path, 'rb') as f:
    meta = orjson.loads(f.read())

output_dict = {}

# Processing the list of JSON objects
for key in meta:
    obj = meta[key]
    if 'raw' in obj:
        # Extract substring from first '{' to the first '}'
        text = obj['raw']
        start = text.find('{')
        end = text.find('}') + 1
        if start != -1 and end != -1:
            try:
                trimmed_text = text[start:end]
                # Parse the trimmed text into JSON
                parsed_text = orjson.loads(trimmed_text)

                # Validate that the parsed JSON has exactly the required keys
                if set(parsed_text.keys()) == required_keys_set:
                    obj["data"] = parsed_text
                    output_dict[key] = obj
                else:
                    print(f"Warning: Parsed JSON in object ID {obj.get('id')} does not match required keys.")
            except ValueError as e:
                print(f"Error parsing 'trimmed_text' in object ID {obj.get('id')}: {e}")

# Save the updated data back to the file
with open(output_path, 'wb') as f:
    f.write(orjson.dumps(output_dict, option=orjson.OPT_INDENT_2))

print(f"Processed data saved to {output_path}")
print(f"The final parsed data was {len(output_dict.keys())}")