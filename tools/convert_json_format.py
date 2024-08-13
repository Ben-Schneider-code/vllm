import orjson
import sys

# Input and output file paths
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Initialize an empty list to hold all JSON objects
json_objects = []

# Read the input file line by line and parse each line as a JSON object
with open(input_file_path, 'r') as file:
    for line in file:
        json_object = orjson.loads(line.strip())
        json_objects.append(json_object)

# Write the list of JSON objects to the output file
with open(output_file_path, 'wb') as output_file:
    # Serialize the list to JSON and write it to the file
    output_file.write(orjson.dumps(json_objects))

print(f"File '{output_file_path}' has been created with all JSON objects as a list.")
