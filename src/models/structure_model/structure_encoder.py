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
from models.structure_model.graphcnn import HierarchyGCN
from models.structure_model.tree import Tree
import json
import os
import numpy as np
from utils.utils import get_hierarchy_relations

class StructureEncoder(nn.Module):
    # GCN 임베딩 
    def __init__(self,
                 config,
                 label_map,
                 device,
                 total_label_map=None,
                 gcn_in_dim=None):
        super(StructureEncoder, self).__init__()

        self.label_map = label_map
        self.root = Tree(-1)
        self.gcn_in_dim = gcn_in_dim if gcn_in_dim is not None else config.structure_encoder.node.dimension
        
        self.hierarchical_label_dict = get_hierarchy_relations(os.path.join(config.data.data_dir, config.data.hierarchy),
                                                                                 self.label_map,
                                                                                 root=self.root,
                                                                                 fortree=False)
        hierarchy_prob_file = os.path.join(config.data.data_dir, config.data.prob_json)
        f = open(hierarchy_prob_file, 'r')
        hierarchy_prob_str = f.readlines()
        f.close()
        self.hierarchy_prob = json.loads(hierarchy_prob_str[0])

        self.node_prob_from_parent = np.zeros((len(self.label_map), len(self.label_map)))
        self.node_prob_from_child = np.zeros((len(self.label_map), len(self.label_map)))

        for p in self.hierarchy_prob.keys():
            if p == 'Root':
                continue
            for c in self.hierarchy_prob[p].keys():
                if p not in self.label_map or c not in self.label_map:
                    continue
                self.node_prob_from_parent[int(self.label_map[c])][int(self.label_map[p])] = self.hierarchy_prob[p][c]
                self.node_prob_from_child[int(self.label_map[p])][int(self.label_map[c])] = 1.0

        self.model = HierarchyGCN(num_nodes=len(self.label_map),
                                    in_matrix=self.node_prob_from_child,
                                    out_matrix=self.node_prob_from_parent,
                                    in_dim=self.gcn_in_dim,
                                    dropout=config.structure_encoder.node.dropout,
                                    device=device,
                                    root=self.root,
                                    hierarchical_label_dict=self.hierarchical_label_dict,
                                    label_trees=None)

    def forward(self, inputs):
        return self.model(inputs)