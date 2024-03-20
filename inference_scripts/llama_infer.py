from datasets import load_from_disk
import json
from vllm import LLM, SamplingParams

dataset = load_from_disk('./alpaca_data')
test_set = dataset['test']

model_path = '/workspace/storage/NLP/1c/finetuned_models/llama2-alpaca'
llm = LLM(model=model_path)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

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

with open('./llm_responses/llama_test_set_predictions.jsonl', 'w') as outfile:
    for example in test_set:
        formatted_prompt = formatting_prompts_func_for_inference(example)
        # Generate response for the formatted prompt
        outputs = llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text  # Assuming one output per prompt
        
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
