import argparse
from datasets import load_from_disk
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# Add command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--output_file", type=str, default="./llm_responses/llama_test_set_predictions.jsonl")
args = parser.parse_args()

dataset = load_from_disk('./alpaca_data')
test_set = dataset['test']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base model and tokenizer
base_model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)

# Move the model to the GPU
model = model.to(device)

# Load the fine-tuned model with LoRA
new_model = "./finetuned_models/phi2/checkpoint-3244"
model = PeftModel.from_pretrained(model, new_model)

# Merge the LoRA parameters with the base model
model = model.merge_and_unload()

# Set padding tokens
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def formatting_prompts_func_for_inference(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    if len(input_text) >= 2:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Input:
        {input_text}
        '''
    else:
        text = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        '''
    return text

with open(args.output_file, 'w') as outfile:
    for example in test_set:
        formatted_prompt = formatting_prompts_func_for_inference(example)
        
        # Tokenize the formatted prompt
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        # Move the input tensors to the GPU
        input_ids = input_ids.to(device)
        
        # Generate response for the formatted prompt
        if(args.num_beams > 1):
            output = model.generate(
                input_ids,
                max_length=256,
                num_return_sequences=1,
                num_beams=args.num_beams,
                early_stopping=True
        )
        elif(args.temperature == 0.0):
            output = model.generate(
                input_ids,
                max_length=256,
                num_return_sequences=1,
                top_k=args.top_k,
                do_sample=False,
                temperature=args.temperature
        )
        else:
            output = model.generate(
                input_ids,
                max_length=256,
                num_return_sequences=1,
                top_k=args.top_k,
                do_sample=True,
                temperature=args.temperature
        )    

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Construct the record to be saved
        record = {
            "prompt": formatted_prompt,
            "generated_text": generated_text,
            "expected_output": example["output"]
        }
        
        # Write the record as a JSON Line
        json.dump(record, outfile)
        outfile.write('\n')

print("Inference complete and results saved.")