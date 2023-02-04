#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
from models.structure_model.structure_encoder import StructureEncoder
from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.text_feature_propagation import HiMatchTP

from transformers import BertPreTrainedModel, BertModel, BertConfig

class HiMatch(nn.Module):
    def __init__(self, config, label_v2i, model_mode='TRAIN'):
        """
        Hierarchy-Aware Global Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_type: Str, ('HiAGM-TP' for the serial variant of text propagation,
                                 'HiAGM-LA' for the parallel variant of multi-label soft attention,
                                 'Origin' without hierarchy-aware module)
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(HiMatch, self).__init__()
        self.config = config
        self.label_v2i = label_v2i
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.dataset = config.data.dataset

        self.model_type = config.model.type
        
        if "bert" in self.model_type:
            model_config = BertConfig.from_pretrained(
                f'{config.model.model_dir}/korscibert_pytorch/model/bert_config_kisti.json',
            )
            self.bert = BertModel.from_pretrained(
                f'{config.model.model_dir}/korscibert_pytorch/model/pytorch_model.bin',
                config=model_config,
            )

            self.bert_dropout = nn.Dropout(0.1)
        
        self.structure_encoder = StructureEncoder(config=config,
                                                label_map=label_v2i,
                                                device=self.device,
                                                graph_model_type=config.structure_encoder.type)
        self.structure_encoder_label = StructureEncoder(config=config,
                                                      label_map=label_v2i,
                                                      device=self.device,
                                                      graph_model_type=config.structure_encoder.type,
                                                      gcn_in_dim=config.embedding.label.dimension)

        self.himatch = HiMatchTP(config=config,
                                 device=self.device,
                                 graph_model=self.structure_encoder,
                                 label_map=self.label_v2i,
                                 graph_model_label=self.structure_encoder_label,
                                 model_mode=model_mode)


    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.himatch.parameters()})
        return params

    def forward(self, inputs):
        """
        forward pass of the overall architecture
        :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return:
        """
        if inputs[1] == "TRAIN":
            batch, mode, label_repre = inputs
            if "bert" in self.model_type:
                outputs = self.bert(batch['input_ids'].to(self.device), batch['segment_ids'].to(self.device), batch['input_mask'].to(self.device))
                pooled_output = outputs[1]
                token_output = self.bert_dropout(pooled_output)
            
            logits, text_repre, label_repre_positive, label_repre_negative = self.himatch(
                [token_output, mode, batch['ranking_positive_mask'], batch['ranking_negative_mask'], label_repre])
            return logits, text_repre, label_repre_positive, label_repre_negative

        else:
            batch, mode = inputs[0], inputs[1]

            if "bert" in self.model_type:
                outputs = self.bert(batch['input_ids'].to(self.device), batch['segment_ids'].to(self.device), batch['input_mask'].to(self.device))
                pooled_output = outputs[1]
                token_output = self.bert_dropout(pooled_output)
            
            logits = self.himatch([token_output, mode])
            return logits
            
    def get_embedding(self, inputs):
        batch, mode = inputs[0], inputs[1]
        if "bert" in self.model_type:
            outputs = self.bert(batch['input_ids'].to(self.device), batch['segment_ids'].to(self.device), batch['input_mask'].to(self.device))
            pooled_output = outputs[1]
            pooled_output = self.bert_dropout(pooled_output)

        return pooled_output
