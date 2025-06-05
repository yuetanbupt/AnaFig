import argparse
import json
import os
from rouge_score import rouge_scorer


def main():
    parser = argparse.ArgumentParser(description='Calculate ROUGE Scores')
    parser.add_argument('--file_name', required=True, help='Input JSON file')
    args = parser.parse_args()

    with open(args.file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []

    for key, value in data.items():
        if 'summary' in value and 'summary_pre' in value and value['summary_pre'] != "error":
            ref = str(value['summary'])
            gen = str(value['summary_pre'])
            scores.append(scorer.score(ref, gen))

    if not scores:
        print("No valid samples for ROUGE")
        return

    # Calculate averages
    avg_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for score in scores:
        for metric in avg_scores:
            avg_scores[metric] += score[metric].fmeasure

    for metric in avg_scores:
        avg_scores[metric] /= len(scores)

    results = {
        "individual_scores": scores,
        "average_scores": avg_scores
    }

    save_results(args.file_name, "rouge", results)
    print(f"Average ROUGE-1: {avg_scores['rouge1']:.4f}")
    print(f"Average ROUGE-2: {avg_scores['rouge2']:.4f}")
    print(f"Average ROUGE-L: {avg_scores['rougeL']:.4f}")


def save_results(input_path, metric, results):
    base_name = os.path.basename(input_path).split('.')[0]
    output_dir = "output/score"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}-{metric}_score.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
