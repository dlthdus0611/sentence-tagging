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

import torch
import numpy as np
import os
from torch import nn
from models.matching_network import MatchingNet
import torch.nn.functional as F

class HiMatchTP(nn.Module):
    # korbert + gcn 을 통해 text를 임베딩하고 classifier를 통해 logits을 뽑음. 
    # matching logits을 뽑기 위해 MatchingNet을 이용해 text와 label의 representatoin을 뽑음. 
     
    def __init__(self, config, args, label_map, encoder, device, model_mode, graph_model_label=None):
        super(HiMatchTP, self).__init__()

        self.config = config
        self.args = args
        self.device = device
        self.label_map = label_map

        self.encoder = encoder
        self.graph_model_label = graph_model_label
        
        self.linear = nn.Linear(len(self.label_map) * config.model.linear_transformation.node_dimension,
                                    len(self.label_map))
        
        self.transformation = nn.Linear(config.model.linear_transformation.text_dimension,
                                            len(self.label_map) * config.model.linear_transformation.node_dimension)

        self.transformation_dropout = nn.Dropout(p=config.model.linear_transformation.dropout)
        self.dropout = nn.Dropout(p=config.model.classifier.dropout)
        self.matching_model = MatchingNet(config, self.graph_model_label, label_map)

    def forward(self, inputs):
        if inputs[1] == "TRAIN":
            text_feature, mode, ranking_positive_mask, ranking_negative_mask, label_repre = inputs
        else:
            text_feature, mode = inputs[0], inputs[1]
            
        text_feature = text_feature.view(text_feature.shape[0], -1)
        text_feature = self.transformation_dropout(self.transformation(text_feature))
        text_feature = text_feature.view(text_feature.shape[0],
                                                 len(self.label_map),
                                                 self.config.model.linear_transformation.node_dimension)

        label_wise_text_feature = self.encoder(text_feature)
            
        logits = self.linear(label_wise_text_feature.contiguous().view(label_wise_text_feature.shape[0], -1))
        if inputs[1] == "TRAIN" and self.config.model.classifier.output_drop:
            logits = self.dropout(logits)

        if inputs[1] == "TRAIN":
            text_repre, label_repre_positive, label_repre_negative = self.matching_model(label_wise_text_feature.contiguous().view(label_wise_text_feature.shape[0], -1),
                                                                                  ranking_positive_mask,
                                                                                  ranking_negative_mask, label_repre)
            return logits, text_repre, label_repre_positive, label_repre_negative
        else:
            return logits

