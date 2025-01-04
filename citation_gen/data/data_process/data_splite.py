import json

with open('../data/citation.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f]
train_data = []
test_data = []
for article in data:
    if article['train_or_test'] == 'train':
        train_data.append({
            'citing_paper_id': article['src_paper_id'],
            'cited_paper_id': article['tgt_paper_id'],
            'citing_paper_abstract': article['src_abstract'],
            'cited_paper_abstract': article['tgt_abstract'],
            'citation': article['explicit_citation'],
            'context': article['text_before_explicit_citation'] + '[Citation]' + article['text_after_explicit_citation']
        })
    elif article['train_or_test'] == 'test':
        test_data.append({
            'citing_paper_id': article['src_paper_id'],
            'cited_paper_id': article['tgt_paper_id'],
            'citing_paper_abstract': article['src_abstract'],
            'cited_paper_abstract': article['tgt_abstract'],
            'citation': article['explicit_citation'],
            'context': article['text_before_explicit_citation'] + '[Citation]' + article['text_after_explicit_citation']
        })

print(len(train_data))
print(len(test_data))

with open('../data/train.json', 'w', encoding='utf-8') as f:
    for article in train_data:
        f.write(json.dumps(article, ensure_ascii=False) + '\n')

with open('../data/test.json', 'w', encoding='utf-8') as f:
    for article in test_data:
        f.write(json.dumps(article, ensure_ascii=False) + '\n')
