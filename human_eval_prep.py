import os
import json
import random

def select_random_samples(file_path, num_samples):
    with open(file_path, 'r') as file:
        responses = [json.loads(line) for line in file]
    
    selected_samples = random.sample(responses, num_samples)
    return selected_samples

def compile_human_evaluation_data(llms, num_samples, output_file):
    human_evaluation_data = []
    
    for llm in llms:
        for file_name in os.listdir("./llm_responses"):
            if file_name.startswith(f"{llm}_cleaned_") and file_name.endswith(".jsonl"):
                file_path = os.path.join("./llm_responses", file_name)
                
                # Extract configuration details from the file name
                config = file_name[len(f"{llm}_cleaned_"):-6]
                
                selected_samples = select_random_samples(file_path, num_samples)
                
                for sample in selected_samples:
                    human_evaluation_data.append({
                        "llm": llm,
                        "config": config,
                        "generated_text": sample["generated_text"],
                        "expected_output": sample["expected_output"]
                    })
    
    with open(output_file, 'w') as file:
        json.dump(human_evaluation_data, file, indent=2)
    
    print(f"Human evaluation data compiled and saved to {output_file}")

# Set the parameters
llms = ["llama", "mistral", "phi"]
num_samples = 5
output_file = "./human_evaluation_data.json"

# Compile human evaluation data
compile_human_evaluation_data(llms, num_samples, output_file)