import os
import orjson

# File path from the environment variable
file_path = os.environ["WIKI_INSTRUCT_ROOT"]
keys = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Answer1", "Answer2", "Answer3", "Answer4"]
required_keys_set = set(keys)

# Reading the file and loading its content
with open(file_path, 'rb') as f:
    meta = orjson.loads(f.read())

# Processing the list of JSON objects
for obj in meta:
    if 'text' in obj:
        # Extract substring from first '{' to the first '}'
        text = obj['text']
        start = text.find('{')
        end = text.find('}') + 1
        if start != -1 and end != -1:
            try:
                trimmed_text = text[start:end]
                # Parse the trimmed text into JSON
                parsed_text = orjson.loads(trimmed_text)

                # Validate that the parsed JSON has exactly the required keys
                if set(parsed_text.keys()) == required_keys_set:
                    obj['parsed_text'] = parsed_text
                else:
                    print(f"Warning: Parsed JSON in object ID {obj.get('id')} does not match required keys.")
            except ValueError as e:
                print(f"Error parsing 'trimmed_text' in object ID {obj.get('id')}: {e}")

# Save the updated data back to the file
with open(file_path, 'wb') as f:
    f.write(orjson.dumps(meta, option=orjson.OPT_INDENT_2))

print(f"Processed data saved to {file_path}")
