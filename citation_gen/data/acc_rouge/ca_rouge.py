from rouge import Rouge
import pandas as pd


def calculate_rouge(data):
    rouge = Rouge()
    rouge_scores = {'rouge-1': 0.0,
                    'rouge-2': 0.0,
                    'rouge-l': 0.0}
    num_samples = len(data)

    for sample in data:
        scores = rouge.get_scores(sample["answer"], sample["responses"])

        for metric in rouge_scores.keys():
            rouge_scores[metric] += scores[0][metric]['f']
    #
    for metric in rouge_scores.keys():
        rouge_scores[metric] /= num_samples

    return rouge_scores


df = pd.read_csv('../result/gpt4_zero_result.csv')
data = df.to_dict(orient='records')

average_scores = calculate_rouge(data)
print(average_scores)
# print("Average ROUGE-1 score:", average_scores['rouge-1']['f'])
# print("Average ROUGE-2 score:", average_scores['rouge-2']['f'])
# print("Average ROUGE-L score:", average_scores['rouge-l']['f'])
