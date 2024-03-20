import os
import torch
from trl import SFTTrainer
from datasets import load_dataset
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
)

base_model = "microsoft/phi-2"

dataset = load_from_disk('./alpaca_data')
dataset = dataset['train']

max_seq_length = 2048

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 2:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Input:
            {input_text}
            
            ### Response:
            {response}
            '''
        else:
            text = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Response:
            {response}
            '''
        output_text.append(text)

    return output_text

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    trust_remote_code=True,
    revision="refs/pr/23" #the main version of Phi-2 doesn’t support gradient checkpointing (while training this model)
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# LoRA configuration
# peft_config = LoraConfig(
#     r=64,                   #default=8
#     lora_alpha= 16,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules= ["Wqkv", "out_proj"] #["Wqkv", "fc1", "fc2" ] # ["Wqkv", "out_proj", "fc1", "fc2" ]
# )
peft_config = LoraConfig(
    r=32, 
    lora_alpha=32, 
    target_modules = [ "q_proj", "k_proj", "v_proj", "dense" ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    formatting_func=formatting_prompts_func,
    max_seq_length= max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
    output_dir = "./finetuned_models/phi2",
    num_train_epochs = 1,
    gradient_checkpointing=True,
    optim = "paged_adamw_32bit",
    save_strategy = "epoch"
    ),
)

trainer.train()

trainer.model.save_pretrained("./finetuned_models/phi2-alpaca")