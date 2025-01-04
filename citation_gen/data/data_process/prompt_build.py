import json
from tqdm import tqdm
import pandas as pd

input_file = '../data/train.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f]

sys_prompt = ("You are a research paper writing assistant, proficient in generating citation sentences based on the user's paper in a given context. I will provide you with the abstract of the citing paper, the abstract of the cited paper, and the context for generating the citation sentence at the [Citation] location within the text. In the citation sentence, #REFR represents the citation of this paper, while #OTHEREFR represents the citation of another paper.The citation sentence should be between 25-30 words.\nThe citing paper refers to the paper the user is writing, the cited paper is the reference paper, and the context refers to the surrounding text for generating the citation sentence. The generated citation sentence needs to be logically coherent within the context, while also reflecting the actual content of the citing paper and the cited paper. The context text will provide the linguistic environment and purpose of the citation sentence, which is crucial to consider when generating it. It's essential to ensure that the generated citation sentence is not only logically sound and linguistically appropriate when combined with the context but also faithfully reflects the core content of both the citing and cited papers.\n###Abstract of the citing paper: {citing_paper_abstract}\n###Abstract of the cited paper: {cited_paper_abstract}\n###Context text: {context}\n###Response:")

prompt_data = []
for article in tqdm(data):
    prompt_data.append({
        'input': sys_prompt.format(**article),
        "output": article['citation']
    })

df = pd.DataFrame(prompt_data)
df.to_csv('../prompt_data/train.csv')

with open('../prompt_data/train.json', 'w', encoding='utf-8') as f:
    for article in prompt_data:
        f.write(json.dumps(article, ensure_ascii=False) + '\n')
