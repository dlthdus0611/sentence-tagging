#!/usr/bin/env python
# coding: utf-8

from torch.utils.data.dataset import Dataset
from models.structure_model.tree import Tree
import helper.logger as logger
import json
import os
import random
import pandas as pd
from helper.utils import get_hierarchy_relations, get_parent, get_sibling
import pretrained_model.korscibert_pytorch.tokenization_kisti as tokenization

class ClassificationDataset(Dataset):
    def __init__(self, config, df, label_v2i, label_i2v, stage='TRAIN'):
        """
        Dataset for text classification based on torch.utils.data.dataset.Dataset
        :param config: helper.configure, Configure Object
        :param stage: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
        """
        super(ClassificationDataset, self).__init__()
        # new
        self.config = config
        # vocab to index, index to vocab
        self.label_v2i = label_v2i
        self.label_i2v = label_i2v

        self.data = df
        self.sentences = self.data['sentence'].values
        self.labels = self.data['total_label'].values

        self.max_input_length = config.text_encoder.max_length
        self.stage = stage
        self.sample_num = config.data.sample_num
        self.negative_ratio = config.data.negative_ratio

        # parent_id to child_id
        self.get_child = get_hierarchy_relations(os.path.join(config.data.data_dir, config.data.hierarchy),
                                                 self.label_v2i,
                                                 root=Tree(-1),
                                                 fortree=False)
        # child_id to parent_id eurlex use file
        self.get_parent = get_parent(self.get_child, config)
        # child_id to sibling_id
        self.get_sibling, self.first_layer = get_sibling(os.path.join(config.data.data_dir, config.data.hierarchy), self.get_child, config, self.label_v2i)
        self.tokenizer = tokenization.FullTokenizer(
                                vocab_file=f'{config.model.model_dir}/korscibert_pytorch/model/vocab_kisti.txt',  
                                do_lower_case=False,  
                                tokenizer_type="Mecab")  

    def __len__(self):
        """
        get the number of samples
        :return: self.corpus_size -> Int
        """
        if self.stage == 'DESC':
            return self.data.shape[0]
        else:
            return self.data.shape[0]

    def __getitem__(self, index):
        """
        sample from the overall corpus
        :param index: int, should be smaller in len(corpus)
        :return: sample -> Dict{'token': List[Str], 'label': List[Str], 'token_len': int}
        """
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
        """
        preprocess each sample with the limitation of maximum length and pad each sample to maximum length
        :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
        :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int, 'positive_sample': List[int], 'negative_sample': List[int], 'margin': List[int]}
        """
        sample = {'label': [], 'positive_sample': [], 'negative_sample': [], 'margin': []}

        # sample_sen
        features = self.create_features(sample_sen, self.max_input_length)
        for (features_k, features_v) in features.items():
            sample[features_k] = features_v
                    
        # sample_labels
        sample['label'] = []
        if self.sample_num > len(sample_labels):
            ranking_true_sample = random.sample(sample_labels, len(sample_labels))
        else:
            ranking_true_sample = random.sample(sample_labels, self.sample_num)
        for v in sample_labels:
            if v not in self.label_v2i.keys():
                # logger.warning('Vocab not in ' + 'label' + ' ' + v)
                print('Vocab not in ' + 'label' + ' ' + v)
            else:
                sample['label'].append(self.label_v2i[v])
            if self.stage == "TRAIN" and v in ranking_true_sample:
                p_int = self.label_v2i[v]

                # its parent (no parent -> use other positive label）
                if p_int in self.first_layer or p_int not in self.get_parent:
                    continue
                else:
                    sample['positive_sample'].append(p_int)
                    parent_list = self.get_parent[p_int]
                    
                n_int0 = random.choice(parent_list)
                sample['negative_sample'].append(n_int0)
                sample['margin'].append(0)

                # its sibling
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

                #random
                n_int2 = random.randint(0, len(self.label_v2i) - 1)
                while self.label_i2v[n_int2] in sample_labels:
                    n_int2 = random.randint(0, len(self.label_v2i) - 1)
                sample['negative_sample'].append(n_int2)
                sample['margin'].append(2)

                #logger.info("p_int:" + self.label_i2v[p_int] + " " + "parent:" + self.label_i2v[n_int0] + " " + "sibling:" + self.label_i2v[n_int1] + " random:" + self.label_i2v[n_int2])

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

