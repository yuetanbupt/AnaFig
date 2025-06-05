import argparse
import json
import os
from bert_score import score


def main():
    parser = argparse.ArgumentParser(description='Calculate BERTScore')
    parser.add_argument('--file_name', required=True, help='Input JSON file')
    args = parser.parse_args()

    with open(args.file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gens, refs = [], []
    for key, value in data.items():
        if 'summary' in value and 'summary_pre' in value and value['summary_pre'] != "error":
            gens.append(str(value['summary_pre']))
            refs.append(str(value['summary']))

    if not gens:
        print("No valid samples for BERTScore")
        return

    _, _, F1 = score(gens, refs, lang='en', device='cuda', batch_size=32)
    avg_score = F1.mean().item()

    output = {
        "individual_scores": F1.tolist(),
        "average_score": avg_score
    }

    save_results(args.file_name, "bertscore", output)
    print(f"Average BERTScore: {avg_score:.4f}")


def save_results(input_path, metric, results):
    base_name = os.path.basename(input_path).split('.')[0]
    output_dir = "output/score"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}-{metric}_score.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
