## 1. Create Environment
```
conda create -n assignment_1c python=3.10 -y
conda activate assignment_1c
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install einops
pip install vllm==0.2.7
```

## 2. Dataset Preprocessing
Create train/test set from Alpaca data:

`python split_dataset.py`

## 3. Fine-Tuning Pre-trained LLMs

Run `python finetuning_scripts/finetune_<MODEL-NAME>.py` to perform finetuning of the LLMs. 

Replace `<MODEL-NAME>` with one of `llama`, `mistral` or `phi`.

## 4. Collect LLM responses

`python inference_scripts/<MODEL-NAME>_infer.py`

## 5. Metric Measurements 



## 6. Hyperparameter Tuning