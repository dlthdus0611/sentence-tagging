'''
@inproceedings{chen-etal-2021-hierarchy,
    title = "Hierarchy-aware Label Semantics Matching Network for Hierarchical Text Classification",
    author = "Chen, Haibin  and Ma, Qianli  and Lin, Zhenxi  and Yan, Jiangyue",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-long.337",
    pages = "4370--4379"
} 
'''

import torch.nn as nn
import torch

class MatchingNet(nn.Module):
    # 라벨 임베딩을 위한 네트워크
    # GCN 으로 구조적 임베딩을 하고 특정 인덱스에 해당하는 라벨의 representation을 추출

    def __init__(self, config, graph_model_label=None, label_map=None):
        super(MatchingNet, self).__init__()
        self.config = config
        self.label_map = label_map
        self.positive_sample_num = config.data.positive_num
        self.negative_sample_num = config.data.negative_ratio * self.positive_sample_num
        self.dimension = config.matching_net.dimension
        self.dropout_ratio = config.matching_net.dropout

        self.embedding_net1 = nn.Sequential(nn.Linear(len(self.label_map) * config.model.linear_transformation.node_dimension, self.dimension),
                                                nn.ReLU(),
                                                nn.Dropout(self.dropout_ratio),
                                                nn.Linear(self.dimension, self.dimension))
        self.label_encoder = nn.Sequential(nn.Linear(config.embedding.label.dimension, self.dimension),
                                               nn.ReLU(),
                                               nn.Dropout(self.dropout_ratio),
                                               nn.Linear(self.dimension, self.dimension))
        self.graph_model = graph_model_label
        self.label_map = label_map

    def forward(self, text, gather_positive, gather_negative, label_repre):
        gather_positive = gather_positive.to(text.device)
        gather_negative = gather_negative.to(text.device)

        label_repre = label_repre.unsqueeze(0)
        label_repre = self.graph_model(label_repre)
        label_repre = label_repre.repeat(text.size(0), 1, 1)
        label_repre = self.label_encoder(label_repre)

        label_repre = torch.nn.functional.normalize(label_repre, dim=-1)
        
        label_positive = torch.gather(label_repre, 1, gather_positive.view(text.size(0), self.positive_sample_num, 1).expand(text.size(0), self.positive_sample_num, label_repre.size(-1)))
        label_negative = torch.gather(label_repre, 1, gather_negative.view(text.size(0), self.negative_sample_num, 1).expand(text.size(0), self.negative_sample_num, label_repre.size(-1)))
        text_encoder = self.embedding_net1(text)
        
        text_encoder = torch.nn.functional.normalize(text_encoder, dim=-1)
        
        return text_encoder, label_positive, label_negative

