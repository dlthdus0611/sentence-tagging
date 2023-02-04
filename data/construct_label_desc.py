import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf_vec = TfidfVectorizer(max_df=0.8, min_df=0.1) 

def get_label_embedding(df_dir, label_dict, output_dir):
	label_list = []
	with open(label_dict) as f:
		for line in f.readlines():
			line = line.rstrip().split('\t')
			label_list.append(line[0])
	label_samples = {}
	label_embedding = {}
	
	df = pd.read_csv(df_dir)
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

	with open(output_dir,'w') as f:
		w = csv.writer(f)
		w.writerow({'sentence','total_label'})
		for label in label_list:
			instance = {"sentence": ' '.join(label_embedding[label]), "total_label": label}
			w.writerow(instance.values())
  
if __name__ == '__main__':
	get_label_embedding("../data/tagging_data.csv", "../data/hierar/label.dict", "../data/exp_label_desc.csv")