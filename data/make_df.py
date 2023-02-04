import os
import csv
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

def load_data(data_path, file_name):
    with open(f'{data_path}/{file_name}.json', 'r') as f:
        json_data = json.load(f)
    return json_data
    
def make_total_label(x):
    if x in ['문제 정의', '가설 설정', '기술 정의']:
        return f'연구 목적.{x}'
    elif x in ['제안 방법', '대상 데이터', '데이터처리', '이론/모형']:
        return  f'연구 방법.{x}'
    else:
        return  f'연구 결과.{x}'

def make_csv(json_data, save_path, file_name):
    os.makedirs(save_path, exist_ok=True)
    data = list(map(lambda x: (x['doc_id'], x['sentence'], x['tag'], x['keysentence']), json_data))
    df = pd.DataFrame(data, columns=['doc_id', 'sentence', 'tag', 'keysentence'])
    df['total_label'] = df['tag'].apply(make_total_label)
    df.to_csv(f'{save_path}/{file_name}.csv', index=False)
    print('Make csv file Done.')
    return df

def make_hierar_prob(save_path, df_train):
    prob = {}
    hierar = ['cat', '연구 결과', '연구 목적', '연구 방법']
    df_train['cat'] = df_train['total_label'].apply(lambda x: x.split('.')[0])

    for h in hierar:
        if h == 'cat':
            prob['Root'] = {}
            sub = df_train['cat'].value_counts(1)
            for k,v in zip(sub.index, sub.values):
                prob['Root'][k] = v
        else:
            prob[f'{h}'] = {}
            df_sub = df_train.query(f"cat == '{h}'")
            sub = df_sub['tag'].value_counts(1)
            for k,v in zip(sub.index, sub.values):
                prob[f'{h}'][k] = v

    with open(f'{save_path}/hierar_prob.json', 'w', encoding='utf-8') as file:
        json.dump(prob, file, ensure_ascii=False)
    print('Make hierar_prob file Done.')

def get_label_embedding(data_dir, vec_max, vec_min):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf_vec = TfidfVectorizer(max_df=vec_max, min_df=vec_min)

    label_list = []
    with open(f'{data_dir}/hierar/label.dict') as f:
        for line in f.readlines():
            line = line.rstrip().split('\t')
            label_list.append(line[0])
    label_samples = {}
    label_embedding = {}
	
    df = pd.read_csv(f'{data_dir}/csv/train.csv')
    for idx, row in df.iterrows():
        labels = row["total_label"].split('.')
        for label in labels:
            if label not in label_samples:
                label_samples[label] = [row['sentence']]
            else:
                label_samples[label].append(row['sentence'])

    # TF-IDF 
    for label in label_list:
        if label not in label_samples:
            word = label.split(' ')
        else:
            corpus = label_samples[label]
            tfidf_matrix = tfidf_vec.fit(corpus)
            sorted(tfidf_matrix.vocabulary_, key=lambda x:x[1], reverse=True)
            word = list(tfidf_matrix.vocabulary_.keys())
        label_embedding[label] = word

    with open(f'{data_dir}/csv/label_desc.csv','w') as f:
        w = csv.writer(f)
        w.writerow({'sentence','total_label'})
        for label in label_list:
            instance = {"sentence": ' '.join(label_embedding[label]), "total_label": label}
            w.writerow(instance.values())

if __name__ == '__main__':
    data_path = './data'
    json_path = f'{data_path}/문장_의미_태깅_데이터셋_저널별비율'
    file_list = ['train', 'dev', 'test']

    for file_name in file_list:
        json_data = load_data(json_path, file_name)
        df = make_csv(json_data, f'{data_path}/csv', file_name)
        if file_name == 'train':
            make_hierar_prob(f'{data_path}/hierar', df)

    get_label_embedding(data_path, 0.8, 0.1)