import os
import orjson

# File path from the environment variable
file_path = os.environ["WIKI_INSTRUCT_ROOT"]

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
            trimmed_text = text[start:end]
            try:
                # Parse the trimmed text into JSON and attach it
                obj['parsed_text'] = orjson.loads(trimmed_text)
            except ValueError as e:
                print(f"Error parsing 'trimmed_text' in object ID {obj.get('id')}: {e}")

# Save the updated data back to the file
with open(file_path, 'wb') as f:
    f.write(orjson.dumps(meta, option=orjson.OPT_INDENT_2))

print(f"Processed data saved to {file_path}")
