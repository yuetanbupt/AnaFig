from bert_score import score
import json

def compute_bertscore_batch(generated_list, reference_list):

    P, R, F1 = score(generated_list, reference_list, lang='en', device='cuda', batch_size=32)
    return F1.tolist()

def average_bertscore(scores_list):
    return sum(scores_list) / len(scores_list) if scores_list else 0

if __name__ == "__main__":
    file_name = input("请输入文件名: ")
    with open(file_name, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    generated_list = []
    reference_list = []

    for key, value in data_set.items():
        if 'summarization' in value and 'summarization_only_figure' in value and value['summarization_only_figure'] != "error":
            generated_list.append(str(value['summarization_only_figure']))
            reference_list.append(str(value['summarization']))

    if generated_list:
        bert_scores = compute_bertscore_batch(generated_list, reference_list)
        avg_score = average_bertscore(bert_scores)
        print(f"Average BERT score: {avg_score}")
    else:
        print("No valid samples to compute BERT scores.")



