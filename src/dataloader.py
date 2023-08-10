import os
import random
import torch
import numpy as np
import pandas as pd
from utils.utils import get_hierarchy_relations, get_parent, get_sibling
import models.pretrained_model.korscibert.tokenization_kisti as tokenization
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from models.structure_model.tree import Tree

class ClassificationDataset(Dataset):
    def __init__(self, config, df, label_v2i, label_i2v, stage='TRAIN'):
        super(ClassificationDataset, self).__init__()
        self.config = config

        self.label_v2i = label_v2i
        self.label_i2v = label_i2v

        self.data = df
        self.sentences = self.data['sentence'].values
        self.labels = self.data['total_label'].values

        self.max_input_length = config.text_encoder.max_length
        self.stage = stage
        self.sample_num = config.data.sample_num
        self.negative_ratio = config.data.negative_ratio

        self.get_child = get_hierarchy_relations(os.path.join(config.data.data_dir, config.data.hierarchy),
                                                 self.label_v2i,
                                                 root=Tree(-1),
                                                 fortree=False)
        self.get_parent = get_parent(self.get_child, config)
        self.get_sibling, self.first_layer = get_sibling(os.path.join(config.data.data_dir, config.data.hierarchy), self.get_child, config, self.label_v2i)
        self.tokenizer = tokenization.FullTokenizer(
                                vocab_file=f'{config.model.model_dir}/korscibert/vocab_kisti.txt',  
                                do_lower_case=False,  
                                tokenizer_type="Mecab")  

    def __len__(self):
        if self.stage == 'DESC':
            return self.data.shape[0]
        else:
            return self.data.shape[0]

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError
        sample_sen = self.sentences[index]
        sample_labels = self.labels[index].split('.')
        return self._preprocess_sample(sample_sen, sample_labels)

    def create_features(self, sentences, max_seq_len=256):
        tokens_a = self.tokenizer.tokenize(sentences)
        tokens_b = None

        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_len - len(input_ids))
        input_len = len(input_ids)

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        feature = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'input_len': input_len}
        return feature
        
    def _preprocess_sample(self, sample_sen, sample_labels):
        # label : 정답 대분류와 소분류 라벨
        # positive sample : 논문 문장의 정답 대분류 라벨 또는 소분류 라벨
        # negative sample : 논문 문장의 오답 대분류 라벨 또는 소분류 라벨 
        # margin : 논문 문장과 parent가 같으면 0, parent는 다르지만 sibling이 같으면 1, parent와 sibling 모두 다르면 2 

        sample = {'label': [], 'positive_sample': [], 'negative_sample': [], 'margin': []}

        features = self.create_features(sample_sen, self.max_input_length)
        for (features_k, features_v) in features.items():
            sample[features_k] = features_v
                    
        sample['label'] = []
        if self.sample_num > len(sample_labels):
            ranking_true_sample = random.sample(sample_labels, len(sample_labels))
        else:
            ranking_true_sample = random.sample(sample_labels, self.sample_num)
        for v in sample_labels:
            if v not in self.label_v2i.keys():
                print('Vocab not in ' + 'label' + ' ' + v)
            else:
                sample['label'].append(self.label_v2i[v])
            if self.stage == "TRAIN" and v in ranking_true_sample:
                p_int = self.label_v2i[v]

                if p_int in self.first_layer or p_int not in self.get_parent:
                    continue
                else:
                    sample['positive_sample'].append(p_int)
                    parent_list = self.get_parent[p_int]
                    
                n_int0 = random.choice(parent_list)
                sample['negative_sample'].append(n_int0)
                sample['margin'].append(0)

                if p_int not in self.get_sibling:
                    n_int1 = random.randint(0, len(self.label_v2i) - 1)
                    while self.label_i2v[n_int1] in sample_labels:
                        n_int1 = random.randint(0, len(self.label_v2i) - 1)
                    sibling_list = [n_int1]
                else:
                    sibling_list = self.get_sibling[p_int]
                n_int1 = random.choice(sibling_list)
                sample['negative_sample'].append(n_int1)
                sample['margin'].append(1)

                n_int2 = random.randint(0, len(self.label_v2i) - 1)
                while self.label_i2v[n_int2] in sample_labels:
                    n_int2 = random.randint(0, len(self.label_v2i) - 1)
                sample['negative_sample'].append(n_int2)
                sample['margin'].append(2)

        if self.stage == "TRAIN" and self.sample_num > len(sample_labels):
            oversize = self.sample_num - len(sample_labels)
            while oversize > 0:
                total_index = list(range(0, len(sample_labels), 1))
                select_index = random.choice(total_index)
                sample['positive_sample'].append(sample['positive_sample'][select_index])
                for pad in range(self.negative_ratio):
                    sample['negative_sample'].append(
                        sample['negative_sample'][self.negative_ratio * select_index + pad])
                    sample['margin'].append(sample['margin'][self.negative_ratio * select_index + pad])
                oversize -= 1

        return sample

class Collator(object):
    def __init__(self, config, mode="TRAIN"):
        super(Collator, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.mode = mode
        self.positive_sample_num = config.data.positive_num
        self.negative_sample_num = config.data.negative_ratio * self.positive_sample_num
        self.negative_ratio = config.data.negative_ratio
        self.label_size = 12

        self.version_higher_11 = True
        version = torch.__version__
        version_num = version.split('.')
        if (int(version_num[0]) == 1 and int(version_num[1]) <= 1) or int(version_num[0]) == 0:
            self.version_higher_11 = False

    def _multi_hot(self, batch_labels):
        batch_size = len(batch_labels)
        max_length = max([len(sample) for sample in batch_labels])
        aligned_batch_labels = []
        for sample_label in batch_labels:
            aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
        aligned_batch_labels = torch.Tensor(aligned_batch_labels).long()
        batch_labels_multi_hot = torch.zeros(batch_size, self.label_size).scatter_(1, aligned_batch_labels, 1)
        return batch_labels_multi_hot

    def __call__(self, batch):
        batch_label = []
        batch_doc_len = []
        batch_ranking_positive_mask = []
        batch_ranking_negative_mask = []
        batch_margin_mask = []

        batch_negative_mask = []
        batch_negative_mask_label = []

        batch_input_ids = []
        batch_input_mask = []
        batch_segment_ids = []
        batch_input_len = []
        
        for sample_i, sample in enumerate(batch):
            batch_label.append(sample['label'])
            if self.mode == "TRAIN":
                positive = np.zeros((self.positive_sample_num))
                negative = np.zeros((self.negative_sample_num))
                margin = np.zeros((self.negative_ratio, self.negative_sample_num))
                for i in range(self.positive_sample_num):
                    positive[i] = sample['positive_sample'][i]
                for i in range(self.negative_sample_num):
                    negative[i] = sample['negative_sample'][i]
                    margin[sample['margin'][i], i] = 1
                batch_ranking_positive_mask.append(positive)
                batch_ranking_negative_mask.append(negative)
                batch_margin_mask.append(margin)
            batch_input_ids.append(sample['input_ids'])
            batch_input_mask.append(sample['input_mask'])
            batch_segment_ids.append(sample['segment_ids'])
            batch_input_len.append(sample['input_len'])

        batch_multi_hot_label = self._multi_hot(batch_label)
        batch_doc_len = torch.FloatTensor(batch_doc_len)

        batch_input_ids = torch.LongTensor(batch_input_ids)
        batch_input_mask = torch.LongTensor(batch_input_mask)
        batch_segment_ids = torch.LongTensor(batch_segment_ids)
        batch_input_len = torch.LongTensor(batch_input_len)
        
        if self.mode == "TRAIN":
            batch_ranking_positive_mask = torch.LongTensor(batch_ranking_positive_mask)
            batch_ranking_negative_mask = torch.LongTensor(batch_ranking_negative_mask)
            batch_margin_mask = torch.BoolTensor(batch_margin_mask)
            batch_res = {
                'label': batch_multi_hot_label,
                'label_list': batch_label,
                'ranking_positive_mask': batch_ranking_positive_mask,
                'ranking_negative_mask': batch_ranking_negative_mask,
                'margin_mask': batch_margin_mask,
                'input_ids': batch_input_ids,
                'input_mask': batch_input_mask,
                'segment_ids': batch_segment_ids,
                'input_len': batch_input_len
            }
        else:
            batch_res = {
                'label': batch_multi_hot_label,
                'label_list': batch_label,
                'input_ids': batch_input_ids,
                'input_mask': batch_input_mask,
                'segment_ids': batch_segment_ids,
                'input_len': batch_input_len
            }
        return batch_res

def data_loaders(config, df, label_v2i, label_i2v, stage="TRAIN"):
    collate_fn_train = Collator(config, mode="TRAIN")
    collate_fn = Collator(config, mode="TEST")

    dataset = ClassificationDataset(config, df, label_v2i, label_i2v, stage=stage)

    if stage == 'TRAIN':
        loader = DataLoader(dataset,
                            batch_size=config.train.batch_size,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=collate_fn_train,
                            pin_memory=True)
    elif stage == 'DESC':
        loader = DataLoader(dataset,
                            batch_size=config.train.batch_size,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=collate_fn,
                            pin_memory=True)
    else:
        loader = DataLoader(dataset,
                            batch_size=config.train.batch_size,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=collate_fn,
                            pin_memory=True)

    return loader
