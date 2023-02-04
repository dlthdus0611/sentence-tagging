from models.model import HiMatch
from utils.train_modules import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, ClassificationLoss, MarginRankingLoss
from utils.utils import *
from utils.evaluation_modules import evaluate_metric, AvgMeter
from dataloader import data_loaders, ClassificationDataset
from sklearn.model_selection import StratifiedKFold
import models.pretrained_model.korscibert.tokenization_kisti as tokenization

import os
import tqdm
import torch
import random
import pickle
import numpy as np
import pandas as pd
from ptflops import get_model_complexity_info
#for debug

def input_constructor(input_res):
    input_ids = torch.ones(()).new_empty((1, input_res[0])).long().cuda()
    segment_ids = torch.ones(()).new_empty((1, input_res[0])).long().cuda()
    input_mask = torch.ones(()).new_empty((1, input_res[0])).long().cuda()
    batch = {'input_ids': input_ids, 'segment_ids':segment_ids, 'input_mask':input_mask}
    mode = 'TEST'
    label_repre = -1
    return {'inputs' : [batch, mode, label_repre]}


class Trainer(object):
    def __init__(self, args, config, save_path):
        super(Trainer, self).__init__()
        self.config = config
        self.args = args
        self.save_path = save_path

        self.global_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        with open(os.path.join(config.data.data_dir, 'hierar/label_v2i.pickle'), 'rb') as f:
            self.label_v2i = pickle.load(f)
        
        with open(os.path.join(config.data.data_dir, 'hierar/label_i2v.pickle'), 'rb') as f:
            self.label_i2v = pickle.load(f)

        # build up model
        self.model = HiMatch(config, args, self.label_v2i, model_mode='TRAIN').to(self.device)
        macs, params = get_model_complexity_info(self.model, (50,), as_strings=True, 
                                            input_constructor=input_constructor, print_per_layer_stat=True, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Define ranking loss
        self.criterion_ranking = MarginRankingLoss(config)
        # Dataloader
        df_train = pd.read_csv(os.path.join(config.data.data_dir, 'csv/train.csv'))
        df_dev = pd.read_csv(os.path.join(config.data.data_dir, 'csv/dev.csv'))
        df_test = pd.read_csv(os.path.join(config.data.data_dir, 'csv/test.csv'))
        df_desc = pd.read_csv(os.path.join(config.data.data_dir,  'csv/label_desc.csv'))

        self.label_desc_loader = data_loaders(config, df_desc, self.label_v2i, self.label_i2v, stage="DESC")
        self.train_loader = data_loaders(config, df_train, self.label_v2i, self.label_i2v, stage="TRAIN")
        self.dev_loader = data_loaders(config, df_dev, self.label_v2i, self.label_i2v, stage="DEV")
        self.test_loader = data_loaders(config, df_test, self.label_v2i, self.label_i2v, stage="TEST")

        self.loader_map = {"TRAIN": self.train_loader, "DEV": self.dev_loader, "TEST": self.test_loader}

        t_total = int(len(self.train_loader) * (config.train.epoch))
        warmup_steps = int(t_total * 0.1)
        no_decay = ['bias', 'LayerNorm.weight']
        
        self.param_optimizer = list(self.model.named_parameters())
        self.optimizer_grouped_parameters = [
                                        {'params': [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': config.train.optimizer.weight_decay},
                                        {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                                        ]

        # Define training objective & optimizer & Scheduler
        self.criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy), self.label_v2i)
        self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=config.train.optimizer.initial_lr, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    def run_eval(self, mode="DEV"):
        self.model.eval()
        valid_loss = AvgMeter()
        predict_probs = []; target_labels = []
        total_loss = 0.0
        label_repre = -1

        for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
            logits = self.model([batch, mode, label_repre])            
            bce_loss = self.criterion(logits, batch['label'].to(self.device))

            total_loss += bce_loss.item()
            valid_loss.update(total_loss, n=len(batch['label']))

            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])

        performance = evaluate_metric(predict_probs,
                                        target_labels,
                                        self.label_v2i, self.label_i2v)
        
        if mode == 'DEV':
            return valid_loss.avg, performance
        elif mode == 'TEST':
            return performance

    def run_train(self, epoch, mode="TRAIN"):
        self.model.train()
        train_loss = AvgMeter()
        predict_probs = []; target_labels = []
        total_loss = 0.0
        label_repre = -1

        for batch_i, batch in enumerate(tqdm.tqdm(self.loader_map[mode])):
            loss = 0
            self.model.zero_grad(set_to_none=True)

            if batch_i == 0 or batch_i % self.config.train.embd_f == 0:
                with torch.no_grad():
                    for batch_label_i, batch_label in enumerate(self.label_desc_loader):
                        label_embedding = self.model.get_embedding([batch_label, mode])
                        if batch_label_i == 0:
                            label_repre = label_embedding
                        else:
                            label_repre = torch.cat([label_repre, label_embedding], 0)
            logits, text_repre, label_repre_positive, label_repre_negative = self.model([batch, mode, label_repre])
                
            bce_loss = self.criterion(logits, batch['label'].to(self.device))
            
            loss += bce_loss
            loss_inter, loss_intra = self.criterion_ranking(text_repre, label_repre_positive, label_repre_negative, batch['margin_mask'].to(self.device))
            loss += loss_inter
            loss += loss_intra

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

        return train_loss.avg

    def train(self, mode='TRAIN'):
        # Logging
        log_file = os.path.join(self.save_path, 'log.log')
        self.logger = get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(self.args)
        
        best_f1 = 0
        early_stopping = 0

        for epoch in range(0, self.config.train.epoch):
            self.logger.info(f'Epoch:[{epoch:03d}/{self.config.train.epoch:03d}]')
            train_loss_avg = self.run_train(epoch, mode="TRAIN")
            valid_loss_avg, valid_performance = self.run_eval(mode="DEV")
            print('>>>>>>>>>>>>>>>>>> valid', valid_performance)

            self.logger.info(f'Train loss:{train_loss_avg:.3f} | Valid loss:{valid_loss_avg:.3f}')

            if best_f1 < valid_performance['macro_f1']:
                best_f1 = valid_performance['macro_f1']
                best_epoch = epoch
                torch.save({'epoch': best_epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(self.save_path, 'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == self.config.train.early_stopping:
                break

        self.logger.info(f'Best F1 Epoch:{best_epoch} | f1:{best_f1:.4f}')

        self.logger.info(f'Training Done.\n')
        self.logger.info(f'Evaluate on Testset.')
        self.best_checkpoint = os.path.join(self.save_path, 'best_model.pth')
        self.test()

    def test(self):
        if not self.args.do_test:
            weights = torch.load((self.best_checkpoint))['state_dict']
        else:
            weights = torch.load(os.path.join(self.save_path, 'best_model.pth'))['state_dict']

        self.model.load_state_dict(weights)
        self.model.eval()
        
        test_performance = self.run_eval(mode='TEST')

        if not self.args.do_test:
            for metric, value in test_performance.items():
                self.logger.info(f'{metric:<10s}: {value:.4f}')
        else:
            for metric, value in test_performance.items():
                print(f'{metric:<10s}: {value:.4f}')
                
    def predict(self, mode='TEST'):
        weights = torch.load(os.path.join(self.save_path, 'best_model.pth'))['state_dict']
        self.model.load_state_dict(weights)
        self.model.eval()
        
        max_seq_len = 300
        print('')
        sentence = input(">>>>>>>>>>>>>>>>>> Paper Sentence: ")

        sample = {'label': [], 'positive_sample': [], 'negative_sample': [], 'margin': []}
        tokenizer = tokenization.FullTokenizer(
                                        vocab_file=f'{self.config.model.model_dir}/korscibert/vocab_kisti.txt',  
                                        do_lower_case=False,  
                                        tokenizer_type="Mecab")  
        tokens_a = tokenizer.tokenize(sentence)

        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_len - len(input_ids))
        input_len = len(input_ids)

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_mask = torch.tensor(input_mask).unsqueeze(0)
        segment_ids = torch.tensor(segment_ids).unsqueeze(0)
        input_len = torch.tensor(input_len).unsqueeze(0)
        
        features = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'input_len': input_len}
        label_repre = -1
        logits = self.model([features, mode, label_repre])
        predict_results = torch.sigmoid(logits).cpu().tolist()

        np_sample_predict = np.array(predict_results, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)

        print(f">>>>>>>>>>>>>>>>>> Tag for Sentence: [{self.label_i2v[sample_predict_descent_idx[0][1]]}]")