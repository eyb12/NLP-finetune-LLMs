## Create Environment
```
conda create -n assignment_1c python=3.10 -y
conda activate assignment_1c
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install einops
pip install vllm==0.2.7
pip install sacrebleu rouge_score bert_score
```

## Some notes
 - The unsloth library is used for finetuning llama and mistral. It uses QLoRA (Quantized Low Rank Adaptation) behind the scenes and significantly reducing training time and VRAM usage.
 - PEFT was used to finetune Phi-2, this method also uses LoRA
 - These PEFT methods only train a much smaller set of parameters to reduce VRAM footprint and speed up training time

Using these methods do have the potential to reduce performance in the final model (compared to full finetuning), but typically this reduction is negligible or in some cases using these methods actually improve performance.

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

Hyperparameter changes used were: 
- `[10, 25, 40, 75]` for `top_k`
- `[2, 3, 5, 10]` for `num_beams`
- `[0.0, 0.25, 0.5, 1.0]` for `temperature`

For each configuration only one of the above is changed and the others are set to default values of `top_k=50`, `num_beams=1`, `temperature=0.8`. Note that `num_beams=1` essentially means no beam search, since beam search and random sampling are not compatible.

This shell script will run all the experiments in a pipeline: Setting the generation hyperparameters, collecting LLM generations and saving them to `./llm_responses`, cleaning output, and finally computing the evaluation metrics which are saved to `./eval_results`.

## Results
Config describes the generation hyperparameters. (CodeBLEU eval metrics ignored as it is not relevant for the text dataset/task)

### Llama2-Alpaca
| Config | BLEU | ROUGE-L | BERTScore | Human  |
|----------|----------|----------|----------|-------|
| top_k_50_num_beams_1_temp_0.0 | 0.104250 | 0.234846 | 0.835651 |  0.87 |
| top_k_50_num_beams_1_temp_0.5 | 0.093281 | 0.224190 | 0.834736 |  0.77 |
| top_k_10_num_beams_1_temp_0.8 | 0.088869 | 0.229596 | 0.842532 |  0.83 |
| top_k_40_num_beams_1_temp_0.8 | 0.087204 | 0.224284 | 0.843406 | 0.77  |
| top_k_50_num_beams_2_temp_0.8 | 0.098968 | 0.234223 | 0.836924 | 0.7  |
| top_k_50_num_beams_1_temp_0.25 | 0.102562 | 0.234169 | 0.834929 |  0.87 |
| top_k_50_num_beams_5_temp_0.8 | 0.113437 | 0.254784 | 0.834861 | 0.6  |
| top_k_50_num_beams_3_temp_0.8 | 0.104220 | 0.243178 | 0.837395 | 0.77  |
| top_k_75_num_beams_1_temp_0.8 | 0.085980 | 0.221993 | 0.844428 | 0.83  |
| top_k_25_num_beams_1_temp_0.8 | 0.086934 | 0.222303 | 0.843334 | 0.77  |
| top_k_50_num_beams_1_temp_1.0 | 0.071234 | 0.209478 | 0.841053 |  0.8 |
| top_k_50_num_beams_10_temp_0.8 | 0.112773 | 0.240020 | 0.782440 |  0.63 |





### Mistral-Alpaca
| Config | BLEU | ROUGE-L | BERTScore | Human |
|----------|----------|----------|----------|---------|
| top_k_50_num_beams_1_temp_0.25 | 0.094395 | 0.223659 | 0.852543 | 0.87  |
| top_k_50_num_beams_1_temp_1.0 | 0.073075 | 0.209348 | 0.856090 |  0.87 |
| top_k_50_num_beams_1_temp_0.5 | 0.090261 | 0.215826 | 0.848454 |  0.83 |
| top_k_40_num_beams_1_temp_0.8 | 0.085135 | 0.221411 | 0.845139 | 0.77  |
| top_k_25_num_beams_1_temp_0.8 | 0.087344 | 0.212703 | 0.846659 | 0.73  |
| top_k_50_num_beams_10_temp_0.8 | 0.126070 | 0.332443 | 0.870765 |  0.87 |
| top_k_50_num_beams_1_temp_0.0 | 0.094592 | 0.229714 | 0.853408 | 0.87  |
| top_k_75_num_beams_1_temp_0.8 | 0.086474 | 0.209671 | 0.847688 |0.77   |
| top_k_10_num_beams_1_temp_0.8 | 0.086329 | 0.224557 | 0.851953 | 0.77  |
| top_k_50_num_beams_3_temp_0.8 | 0.119682 | 0.294829 | 0.868226 | 0.83  |
| top_k_50_num_beams_5_temp_0.8 | 0.121952 | 0.313187 | 0.871245 |  0.83 |
| top_k_50_num_beams_2_temp_0.8 | 0.112313 | 0.282406 | 0.865394 | 0.8  |

### Phi2-Alpaca
| Config | BLEU | ROUGE-L | BERTScore | Human |
|----------|----------|----------|----------|---------|
| top_k_50_num_beams_1_temp_1.0 | 0.032650 | 0.139256 | 0.778045 | 0.6  |
| top_k_25_num_beams_1_temp_0.8 | 0.040128 | 0.152024 | 0.776562 | 0.7  |
| top_k_50_num_beams_3_temp_0.8 | 0.042344 | 0.153522 | 0.768393 | 0.6  |
| top_k_50_num_beams_10_temp_0.8 | 0.059236 | 0.182520 | 0.775320 | 0.5  |
| top_k_50_num_beams_2_temp_0.8 | 0.041136 | 0.149731 | 0.765922 | 0.6  |
| top_k_50_num_beams_5_temp_0.8 | 0.049616 | 0.169586 | 0.771494 | 0.6  |
| top_k_40_num_beams_1_temp_0.8 | 0.035122 | 0.146049 | 0.774651 | 0.7  |
| top_k_10_num_beams_1_temp_0.8 | 0.037112 | 0.150133 | 0.775434 | 0.6  |
| top_k_50_num_beams_1_temp_0.5 | 0.036085 | 0.146374 | 0.767870 | 0.7  |
| top_k_75_num_beams_1_temp_0.8 | 0.032411 | 0.143313 | 0.769573 |  0.8 |
| top_k_50_num_beams_1_temp_0.25 | 0.037701 | 0.144081 | 0.764685 |  0.6 |

## Discussion Section

**Write a discussion explaining the comparison between two models. Moreover, compare the metrics and discuss which metrics are more appropriate compared to human evaluation.**

Mistral-Alpaca generally outperforms both Llama2-Alpaca and Phi2-Alpaca across all metrics. Llama2-Alpaca shows the second-best performance, while Phi2-Alpaca is behind the other two models. BLEU and ROUGE-L focus on n-gram overlap and are more sensitive to exact word matches, while BERTScore captures semantic similarity using contextualized embeddings. These automated metrics, especially BLEU and ROUGE-L, may not be comprehensive in demonstrating the quality and correctness of responses, but the BERTScore and human evaluations also support the ranking of the LLMs.


**Write another discussion explaining the how the hyperparameters effect on the different metrics of LLaMA, Mistral, Phi-2.**

 - For Llama2-Alpaca, increasing the number of beams (num_beams) tends to improve the BLEU and ROUGE-L scores, with the highest scores achieved when num_beams is set to 5 or 10. However, the BERTScore slightly decreases with higher num_beams values. Varying the temperature or the top_k value does not show a consistent trend in the metrics.

 - In the case of Mistral-Alpaca, higher num_beams values (3, 5, and 10) result in significant improvements across all three metrics compared to the default setting (num_beams=1). Increasing the temperature leads to lower BLEU and ROUGE-L scores, while the BERTScore remains relatively stable. The top_k value does not have a notable impact on the metrics.

 - For Phi2-Alpaca, increasing the num_beams value generally improves the BLEU and ROUGE-L scores, with the highest scores obtained when num_beams is set to 10. However, the BERTScore does not show a clear trend with varying num_beams. Changing the temperature or the top_k value does not result in significant differences in the metrics.

 Overall, the effect of hyperparameters on the metrics varies across the three models. Mistral-Alpaca seems to benefit the most from increasing the num_beams value, while Llama2-Alpaca and Phi2-Alpaca show some improvements but to a lesser extent. The impact of temperature and top_k is less pronounced and inconsistent across the models.


## References and Useful Resources Used
1. [unsloth](https://github.com/unslothai/unsloth) library for faster finetuning with less VRAM footprint (via QLoRA)

2. Supervised fine-tuning trainer to streamline finetuning process, [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)

3. [vLLM](https://github.com/vllm-project/vllm) for fast and memory-efficient inferencing
