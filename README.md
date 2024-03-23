## Create Environment
```
conda create -n assignment_1c python=3.10 -y
conda activate assignment_1c
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install einops
pip install vllm==0.2.7
pip install sacrebleu rogue_score bert_score
```

Alternatively, install using `conda env create -f environment.yaml`

## Some notes
 - The unsloth library is used for finetuning llama and mistral. It uses QLoRA (Quantized Low Rank Adaptation) behind the scenes and significantly reducing training time and VRAM usage.
 - PEFT was used to finetune Phi-2, this method also uses LoRA

Using these methods do have the potential to reduce performance in the final model, but typically this reduction is negligible or in some cases using these methods actually improve performance.

## Dataset Preprocessing
Create train/test set from Alpaca data (as it does not originally have a test split, which we need for evaluations):

`python split_dataset.py`

This will create a folder `./alpaca_data` which stores the train and test split that is created.

## Fine-Tuning Pre-trained LLMs

Run `python finetuning_scripts/finetune_<MODEL-NAME>.py` to perform finetuning of the LLMs. 

Replace `<MODEL-NAME>` with one of `llama`, `mistral` or `phi`. Models will be saved in `./finetuned_models`

## Collect LLM responses and perform metric evaluations

Run `sh hyperparameter_experiments.sh`

This shell script will use 4 different hyperparameter configurations for model generation for each of `top_k`, `num_beams`, and `temperature` (for a total of 12 configurations). Then each configuration will be applied to each of the 3 LLMs: Llama2-7B, Mistral-7B, and Phi-2-2.7B.

This shell script will run all the experiments in a pipeline: Setting the generation hyperparameters, collecting LLM generations and saving them to `./llm_responses`, cleaning output, and finally computing the evaluation metrics which are saved to `./eval_results`.

## Results

### Llama2-Alpaca
|Config    | BLEU     | ROUGE-L  | BERTScore| 
|----------|----------|----------|----------|
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 




### Mistral-Alpaca
|Config    | BLEU     | ROUGE-L  | BERTScore| 
|----------|----------|----------|----------|
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 

### Phi2-Alpaca
|Config    | BLEU     | ROUGE-L  | BERTScore| 
|----------|----------|----------|----------|
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 
|          |          |          |          | 

## Discussion Section

Write a discussion explaining the comparison between two models. Moreover, compare the metrics and discuss which metrics are more appropriate compared to human evaluation.



Write another discussion explaining the how the hyperparameters effect on the different metrics of LLaMA, Mistral, Phi-2.

