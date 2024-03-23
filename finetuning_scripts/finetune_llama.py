from datasets import load_dataset
from datasets import load_from_disk
import transformers
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

dataset = load_from_disk('./alpaca_data')
train_set = dataset['train']

max_seq_length = 256

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    token = "HF_TOKEN HERE"
)

EOS_TOKEN = tokenizer.eos_token
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
        
        text = text + EOS_TOKEN
        output_text.append(text)

    return output_text


# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 4289,
    max_seq_length = max_seq_length,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = train_set,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    formatting_func = formatting_prompts_func,
    args = TrainingArguments(
      num_train_epochs=1,
      fp16 = not torch.cuda.is_bf16_supported(),
      bf16 = torch.cuda.is_bf16_supported(),
      output_dir = "./finetuned_models/llama",
      optim = "adamw_8bit",
      save_strategy = "epoch",
      seed = 4289,
    ),
)

trainer.train()

trainer.model.save_pretrained_merged("./finetuned_models/llama2-alpaca", tokenizer, save_method = "merged_16bit",)
