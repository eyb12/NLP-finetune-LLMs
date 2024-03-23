import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="")
parser.add_argument("--output_file", type=str, default="")
args = parser.parse_args()

# Specify the input and output file paths
input_file = args.input_file
output_file = args.output_file

# Open the input and output JSONL files
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Iterate over each line in the input file
    for line in infile:
        # Parse the JSON object from the line
        data = json.loads(line)
        
        # Get the "generated_text" field
        generated_text = data['generated_text']
        
        # Remove the unwanted characters and strings
        generated_text = generated_text.replace('\\n', ' ')
        generated_text = generated_text.replace('### Response:', '')
        generated_text = generated_text.replace('### Instruction:', '')
        generated_text = generated_text.replace('### Input:', '')
        
        # Update the "generated_text" field in the JSON object
        data['generated_text'] = generated_text.strip()
        
        # Write the updated JSON object to the output file
        outfile.write(json.dumps(data) + '\n')