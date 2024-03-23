import os

def read_metrics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        bleu_score = float(lines[0].split(':')[1].strip())
        rouge_l_score = float(lines[1].split(':')[1].strip())
        bertscore_f1 = float(lines[2].split(':')[1].strip())
        return bleu_score, rouge_l_score, bertscore_f1

def generate_markdown_table(llm_name, metrics):
    table = f"| Config | BLEU | ROUGE-L | BERTScore |\n"
    table += f"|----------|----------|----------|----------|\n"

    for config, scores in metrics.items():
        bleu, rouge_l, bertscore_f1 = scores
        table += f"| {config} | {bleu:.6f} | {rouge_l:.6f} | {bertscore_f1:.6f} |\n"

    return table

def save_markdown_table(llm_name, table):
    output_dir = './eval_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{llm_name}_metrics_table.md")
    with open(file_path, 'w') as file:
        file.write(table)

def main():
    llms = ['llama', 'mistral', 'phi']
    metrics_dir = './eval_results'

    for llm in llms:
        metrics = {}
        for file_name in os.listdir(metrics_dir):
            if file_name.startswith(f"{llm}_metrics_"):
                config = file_name[len(f"{llm}_metrics_"):-4]
                file_path = os.path.join(metrics_dir, file_name)
                bleu, rouge_l, bertscore_f1 = read_metrics(file_path)
                metrics[config] = (bleu, rouge_l, bertscore_f1)

        table = generate_markdown_table(llm, metrics)
        save_markdown_table(llm, table)
        print(f"Markdown table for {llm.upper()} saved.")

if __name__ == '__main__':
    main()