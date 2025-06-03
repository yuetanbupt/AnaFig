
import json
from rouge_score import rouge_scorer

def compute_rouge(model_generated, golden_sum):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    if isinstance(model_generated, list):
        model_generated = " ".join(model_generated)
    if isinstance(golden_sum, list):
        golden_sum = " ".join(golden_sum)

    scores = scorer.score(golden_sum, model_generated)
    return scores

def average_rouge(scores_list):

    avg_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for score in scores_list:
        avg_rouge['rouge1'] += score['rouge1'].fmeasure
        avg_rouge['rouge2'] += score['rouge2'].fmeasure
        avg_rouge['rougeL'] += score['rougeL'].fmeasure

    # 平均化
    num_samples = len(scores_list)
    if num_samples > 0:
        avg_rouge['rouge1'] /= num_samples
        avg_rouge['rouge2'] /= num_samples
        avg_rouge['rougeL'] /= num_samples

    return avg_rouge

if __name__ == "__main__":
    i = 0
    valid_samples = 0  
    scores_list = []  
    
    fileName = input("请输入文件名")
    with open(fileName, "r", encoding="utf-8") as f:
        dataSet = json.load(f)

    for key, value in dataSet.items():
        i += 1   
        if 'summarization' not in value:
            print(f"Skipping sample {key} due to missing 'summarization'.")
            continue  
        G_sum = value['summarization']  
        
        if 'summarization_pre' not in value:
    
            print(f"Skipping sample {key} due to missing 'summarization_pre'.")
            continue  
        if value['summarization_pre'] == "error":

            print(f"Skipping sample {key} due to error content.")
            continue  

        M_sum = value['summarization_pre']
    

        rouge_scores = compute_rouge(M_sum, G_sum)
        scores_list.append(rouge_scores)
        valid_samples += 1  
        print(valid_samples)

  
    if valid_samples > 0:
        avg_scores = average_rouge(scores_list)
        print("Average ROUGE scores:", avg_scores)
    else:
        print("No valid samples to compute ROUGE scores.")
