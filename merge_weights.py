from unsloth import FastLanguageModel

model_path = "./finetuned_models/mistral/checkpoint-6488"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
    )

model.save_pretrained_merged("mistral7b-alpaca", tokenizer, save_method = "merged_16bit",)