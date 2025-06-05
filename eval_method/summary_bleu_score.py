import argparse
import json
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


def main():
    parser = argparse.ArgumentParser(description='Calculate BLEU Score')
    parser.add_argument('--file_name', required=True, help='Input JSON file')
    args = parser.parse_args()

    with open(args.file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scores = []
    for key, value in data.items():
        if 'summary' in value and 'summary_pre' in value and value['summary_pre'] != "error":
            ref = word_tokenize(str(value['summary']))
            gen = word_tokenize(str(value['summary_pre']))
            scores.append(sentence_bleu([ref], gen))

    if not scores:
        print("No valid samples for BLEU")
        return

    avg_score = sum(scores) / len(scores)
    results = {
        "individual_scores": scores,
        "average_score": avg_score
    }

    save_results(args.file_name, "bleu", results)
    print(f"Average BLEU: {avg_score:.4f}")


def save_results(input_path, metric, results):
    base_name = os.path.basename(input_path).split('.')[0]
    output_dir = "output/score"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}-{metric}_score.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
