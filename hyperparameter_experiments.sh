#!/bin/bash

# Function to run inference for an LLM
run_inference() {
    local llm=$1
    local top_k=$2
    local num_beams=$3
    local temperature=$4
    
    # Run the inference script for the LLM with the provided hyperparameters
    python inference_scripts/${llm}_infer.py --top_k ${top_k} --num_beams ${num_beams} --temperature ${temperature} --output_file ./llm_responses/${llm}_generated_top_k_${top_k}_num_beams_${num_beams}_temp_${temperature}.jsonl
    
    # Run the cleaning script on the generated JSONL file
    python clean_responses.py --input_file ./llm_responses/${llm}_generated_top_k_${top_k}_num_beams_${num_beams}_temp_${temperature}.jsonl --output_file ./llm_responses/${llm}_cleaned_top_k_${top_k}_num_beams_${num_beams}_temp_${temperature}.jsonl
    
    # Compute metrics for the cleaned JSONL file
    python compute_metrics.py --input_file ./llm_responses/${llm}_cleaned_top_k_${top_k}_num_beams_${num_beams}_temp_${temperature}.jsonl --output_file ./eval_results/${llm}_metrics_top_k_${top_k}_num_beams_${num_beams}_temp_${temperature}.txt
    
    # Print the completion message for the current iteration
    echo "Completed iteration for ${llm} with top_k=${top_k}, num_beams=${num_beams}, temperature=${temperature}"
}

# Set the default values
default_top_k=50
default_temperature=0.8

# Set the hyperparameter values to test
top_k_values=(10 25 40 75)
num_beams_values=(2 3 5 10)
temperature_values=(0 0.25 0.5 1.0)


# Run experiments for different top_k values
for top_k in "${top_k_values[@]}"
do
    for llm in "llama" "mistral" "phi"
    do
        echo "Running iteration for ${llm} with top_k=${top_k}, num_beams=1, temperature=${default_temperature}"
        run_inference "${llm}" ${top_k} 1 ${default_temperature}
    done
done

# Run experiments for different num_beams values
for num_beams in "${num_beams_values[@]}"
do
    for llm in "llama" "mistral" "phi"
    do
        echo "Running iteration for ${llm} with top_k=${default_top_k}, num_beams=${num_beams}, temperature=${default_temperature}"
        run_inference "${llm}" ${default_top_k} ${num_beams} ${default_temperature}
    done
done

# Run experiments for different temperature values
for temperature in "${temperature_values[@]}"
do
    for llm in "llama" "mistral" "phi"
    do
        echo "Running iteration for ${llm} with top_k=${default_top_k}, num_beams=1, temperature=${temperature}"
        run_inference "${llm}" ${default_top_k} 1 ${temperature}
    done
done