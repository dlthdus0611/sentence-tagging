#!/usr/bin/env python
# coding:utf-8

from data_modules.dataset import ClassificationDataset
from data_modules.collator import Collator
from torch.utils.data import DataLoader
import pandas as pd


def data_loaders(config, df, label_v2i, label_i2v, stage="TRAIN"):
    """
    get data loaders for training and evaluation
    :param config: helper.configure, Configure Object
    :return: -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    
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
