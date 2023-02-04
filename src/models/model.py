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
import torch.nn as nn
from models.structure_model.structure_encoder import StructureEncoder
from models.text_feature_propagation import HiMatchTP
from transformers import BertModel, BertConfig

class HiMatch(nn.Module):
    def __init__(self, config, args, label_v2i, model_mode='TRAIN'):
        super(HiMatch, self).__init__()
        self.config = config
        self.args = args
        self.label_v2i = label_v2i
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        
        model_config = BertConfig.from_pretrained(
                f'{config.model.model_dir}/korscibert/bert_config_kisti.json',
            )
        self.bert = BertModel.from_pretrained(
                f'{config.model.model_dir}/korscibert/pytorch_model.bin',
                config=model_config,
            )

        self.bert_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.pooler.dense.out_features, len(self.label_v2i)) 
        self.encoder = StructureEncoder(config=config,
                                            label_map=label_v2i,
                                            device=self.device)

        self.structure_encoder_label = StructureEncoder(config=config,
                                                        label_map=label_v2i,
                                                        device=self.device,
                                                        gcn_in_dim=config.embedding.label.dimension)
        self.himatch = HiMatchTP(config=config,
                                args=args,
                                device=self.device,
                                encoder=self.encoder,
                                label_map=self.label_v2i,
                                graph_model_label=self.structure_encoder_label,
                                model_mode=model_mode)

    def optimize_params_dict(self):
        params = list()
        params.append({'params': self.himatch.parameters()})
        return params

    def forward(self, inputs):
        # logits : 태깅 분류를 위한 logits
        # text_repre : 입력 논문 문장의 representation
        # label_repre_positive : positive label representation
        # label_repre_negatvie : negative label representation
        
        if inputs[1] == "TRAIN":
            batch, mode, label_repre = inputs
            
            # korbert 임베딩
            outputs = self.bert(batch['input_ids'].to(self.device), batch['segment_ids'].to(self.device), batch['input_mask'].to(self.device))
            pooled_output = outputs[1]
            token_output = self.bert_dropout(pooled_output)

            # 제안하는 모델
            logits, text_repre, label_repre_positive, label_repre_negative = self.himatch(
                                    [token_output, mode, batch['ranking_positive_mask'], batch['ranking_negative_mask'], label_repre])
            
            return logits, text_repre, label_repre_positive, label_repre_negative

        else:
            batch, mode = inputs[0], inputs[1]
            outputs = self.bert(batch['input_ids'].to(self.device), batch['segment_ids'].to(self.device), batch['input_mask'].to(self.device))
            pooled_output = outputs[1]
            token_output = self.bert_dropout(pooled_output)
            logits = self.himatch([token_output, mode])
            
            return logits
            
    def get_embedding(self, inputs):
        batch, mode = inputs[0], inputs[1]
        outputs = self.bert(batch['input_ids'].to(self.device), batch['segment_ids'].to(self.device), batch['input_mask'].to(self.device))
        pooled_output = outputs[1]
        bert_out = self.bert_dropout(pooled_output)

        return bert_out
