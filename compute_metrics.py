import json
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="")
parser.add_argument("--output_file", type=str, default="")
args = parser.parse_args()

def compute_metrics(prediction_file, output_file):
    with open(prediction_file, 'r') as file:
        predictions = [json.loads(line) for line in file]

    generated_texts = [pred['generated_text'] for pred in predictions]
    expected_outputs = [pred['expected_output'] for pred in predictions]

    # Compute BLEU score
    bleu_score = corpus_bleu(generated_texts, [expected_outputs])
    bleu_score_normalized = bleu_score.score / 100.0

    # Compute ROUGE-L score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, hyp) for ref, hyp in zip(expected_outputs, generated_texts)]
    rouge_l_scores = [score['rougeL'].fmeasure for score in rouge_scores]
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    # Compute BERTScore (F1 only)
    _, _, f1 = score(generated_texts, expected_outputs, lang='en', verbose=True)
    avg_f1 = f1.mean().item()

    # Save results to a text file with increased precision
    with open(output_file, 'w') as file:
        file.write(f"BLEU score: {bleu_score_normalized:.6f}\n")
        file.write(f"Average ROUGE-L score: {avg_rouge_l:.6f}\n")
        file.write(f"BERTScore - F1: {avg_f1:.6f}\n")

    print(f"Metrics saved to {output_file}")

# Specify the path to your prediction JSONL file and output text file
prediction_file = args.input_file
output_file = args.output_file

# Compute metrics and save results
compute_metrics(prediction_file, output_file)