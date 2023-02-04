#!/usr/bin/env python
# coding:utf-8

import utils
from utils import *
from models.model import HiMatch
from helper.adamw import AdamW
# import helper.logger as logger
from helper.utils import load_checkpoint, save_checkpoint
from helper.lr_schedulers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from train_modules.evaluation_metrics import evaluate_layer
from data_modules.data_loader import data_loaders
from sklearn.model_selection import StratifiedKFold
from train_modules.criterions import ClassificationLoss, MarginRankingLoss

import os
import tqdm
import torch
import random
import pickle
import neptune
import numpy as np
import pandas as pd

#for debug
torch.autograd.set_detect_anomaly(True)


class Trainer(object):
    def __init__(self, args, config, info):
        super(Trainer, self).__init__()
        
        # Logging
        log_file = os.path.join(info['output_dir'], 'log.log')
        self.logger = utils.get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        self.logger.info(args.tag)

        self.config = config
        self.args = args
        self.info = info

        self.global_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        with open(os.path.join(config.data.data_dir, 'hierar/label_v2i.pickle'), 'rb') as f:
            self.label_v2i = pickle.load(f)
        
        with open(os.path.join(config.data.data_dir, 'hierar/label_i2v.pickle'), 'rb') as f:
            self.label_i2v = pickle.load(f)

        # build up model
        self.model = HiMatch(config, self.label_v2i, model_mode='TRAIN').to(self.device)

        # Define ranking loss
        self.criterion_ranking = MarginRankingLoss(config)

        df_train = pd.read_csv(os.path.join(config.data.data_dir, config.data.train_file))
        kf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=config.train.seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y=df_train['tag'])):
            df_train.loc[val_idx, 'fold'] = fold
        val_idx = list(df_train[df_train['fold'] == int(args.fold)].index)

        df_val = df_train[df_train['fold'] == args.fold].reset_index(drop=True)
        df_train = df_train[df_train['fold'] != args.fold].reset_index(drop=True)
        df_test = pd.read_csv(os.path.join(config.data.data_dir, config.data.test_file))
        df_desc = pd.read_csv(os.path.join(config.data.data_dir, config.data.label_desc_file))
        

        self.label_desc_loader = data_loaders(config, df_desc, self.label_v2i, self.label_i2v, stage="DESC")
        self.train_loader = data_loaders(config, df_train, self.label_v2i, self.label_i2v, stage="TRAIN")
        self.dev_loader = data_loaders(config, df_val, self.label_v2i, self.label_i2v, stage="DEV")
        self.test_loader = data_loaders(config, df_test, self.label_v2i, self.label_i2v, stage="TEST")

        self.loader_map = {"TRAIN": self.train_loader, "DEV": self.dev_loader, "TEST": self.test_loader}

        t_total = int(len(self.train_loader) * (args.epoch))
        warmup_steps = int(t_total * 0.1)
        no_decay = ['bias', 'LayerNorm.weight']
        
        self.param_optimizer = list(self.model.named_parameters())
        self.optimizer_grouped_parameters = [
                                        {'params': [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
                                        {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                                        ]

        # Define training objective & optimizer & Scheduler
        self.criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                        self.label_v2i,
                                        recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                        recursive_constraint=config.train.loss.recursive_regularization.flag, 
                                        loss_type="bce")
        self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=args.initial_lr, eps=1e-8)

        # iter_per_epoch = len(self.train_loader)
        # self.warmup_scheduler = utils.WarmUpLR(self.optimizer, iter_per_epoch * args.warm_epoch)

        if args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone, gamma=args.lr_factor, verbose=True)
        elif args.scheduler == 'cos':
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total) 
        else:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    def run_eval(self, mode="DEV"):
        self.model.eval()
        valid_loss = utils.AvgMeter()
        predict_probs = []; target_labels = []
        total_loss = 0.0
        label_repre = -1

        for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
            logits = self.model([batch, mode, label_repre])
            
            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.hiagm.linear.weight
            else:
                recursive_constrained_params = None

            bce_loss = self.criterion(logits,
                        batch['label'].to(self.device),
                        recursive_constrained_params)

            total_loss += bce_loss.item()
            valid_loss.update(total_loss, n=len(batch['label']))

            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])

        performance = evaluate_layer(predict_probs,
                                        target_labels,
                                        self.label_v2i, self.label_i2v,
                                        self.args.threshold)
        if mode == 'DEV':
            return valid_loss.avg, performance
        elif mode == 'TEST':
            return performance

    def run_train(self, epoch, mode="TRAIN"):
        self.model.train()
        train_loss = utils.AvgMeter()
        predict_probs = []; target_labels = []
        total_loss = 0.0
        label_repre = -1

        for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
            # if epoch <= self.args.warm_epoch:
            #     self.warmup_scheduler.step()
            self.model.zero_grad(set_to_none=True)

            if batch_i == 0 or batch_i % 50 == 0:
                with torch.no_grad():
                    for batch_label_i, batch_label in enumerate(self.label_desc_loader):
                        label_embedding = self.model.get_embedding([batch_label, mode])
                        if batch_label_i == 0:
                            label_repre = label_embedding
                        else:
                            label_repre = torch.cat([label_repre, label_embedding], 0)
            logits, text_repre, label_repre_positive, label_repre_negative = self.model([batch, mode, label_repre])

            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.hiagm.linear.weight
            else:
                recursive_constrained_params = None

            bce_loss = self.criterion(logits,
                                    batch['label'].to(self.device),
                                    recursive_constrained_params)

            loss_inter, loss_intra = self.criterion_ranking(text_repre, label_repre_positive, label_repre_negative, batch['margin_mask'].to(self.device))
            loss = bce_loss + loss_inter + loss_intra
            loss.backward(retain_graph=True)

            total_loss += loss.item()
            train_loss.update(total_loss, n=len(batch['label']))

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])
            # performance = evaluate_layer(predict_probs,
            #                                 target_labels,
            #                                 self.label_v2i, self.label_i2v,
            #                                 self.args.threshold)
            performance = None
        return train_loss.avg, performance

    def train(self, mode='TRAIN'):
        best_f1 = 0
        early_stopping = 0

        for epoch in range(0, self.args.epoch):

            # if epoch > self.args.warm_epoch:
            #     self.scheduler.step()

            self.logger.info(f'Epoch:[{epoch:03d}/{self.args.epoch:03d}]')
            train_loss_avg, train_performance = self.run_train(epoch, mode="TRAIN")
            valid_loss_avg, valid_performance = self.run_eval(mode="DEV")
            # print('>>>>>>>>>>>>>>>>>> train', train_performance)
            print('>>>>>>>>>>>>>>>>>> valid', valid_performance)

            if self.args.logging:
                neptune.log_metric('Train loss', train_loss_avg)
                neptune.log_metric('val loss', valid_loss_avg)

                # for metric, value in train_performance.items():
                #     neptune.log_metric(f'Train {metric}', value)
                for metric, value in valid_performance.items():
                    neptune.log_metric(f'val {metric}', value)

            self.logger.info(f'Train loss:{train_loss_avg:.3f} | Valid loss:{valid_loss_avg:.3f}')

            if best_f1 < valid_performance['l2_macro_f1']:
                best_f1 = valid_performance['l2_macro_f1']
                f1_best_epoch = epoch

                torch.save({'epoch': f1_best_epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(self.info['output_dir'], 'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{f1_best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == self.config.train.early_stopping:
                break

        self.logger.info(f'Best F1 Epoch:{f1_best_epoch} | f1:{best_f1:.4f}')

        self.logger.info(f'Training Done.\n')
        self.logger.info(f'Evaluate on Testset.')
        self.best_checkpoint = os.path.join(self.info['output_dir'], 'best_model.pth')
        self.test()

    def test(self):
        # 원본 문장 가져오기
        if not self.args.do_test:
            weights = torch.load((self.best_checkpoint))['state_dict']
        else:
            weights = torch.load(os.path.join(self.info['test_path'], 'best_model.pth'))['state_dict']

        self.model.load_state_dict(weights)
        self.model.eval()

        # 평가 데이터에 대한 정확도와 모델의 입력, 출력, 정답
        test_performance = self.run_eval(mode='TEST')

        for metric, value in test_performance.items():
            self.logger.info(f'{metric:<10s}: {value:.4f}')
            if self.args.logging:
                neptune.log_metric(f'Test {metric}', value)