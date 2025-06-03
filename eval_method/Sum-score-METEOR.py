import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import json

def compute_meteor(generated, reference):

    generated_tokens = word_tokenize(generated)
    reference_tokens = word_tokenize(reference)
    

    return meteor_score([reference_tokens], generated_tokens)

def average_meteor(scores_list):

    return sum(scores_list) / len(scores_list) if len(scores_list) > 0 else 0

if __name__ == "__main__":
    i = 0
    valid_samples = 0  
    scores_list = []  
    fileName = input("请输入要测试的文件名")
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
        #if value['summarization_only_figure'] == "error":
            print(f"Skipping sample {key} due to error content.")
            continue 
    

        M_sum = str(value['summarization_pre'])
        meteor_score_value = compute_meteor(M_sum, G_sum)
        scores_list.append(meteor_score_value)
        valid_samples += 1  
        print(valid_samples)

    if valid_samples > 0:
        avg_meteor = average_meteor(scores_list)
        print(f"Average METEOR score: {avg_meteor}")
    else:
        print("No valid samples to compute METEOR scores.")

