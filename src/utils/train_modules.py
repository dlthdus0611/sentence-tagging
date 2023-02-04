import torch
import math
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1]  < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss
        
class ClassificationLoss(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map):
        super(ClassificationLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        device = logits.device
        loss = self.loss_fn(logits, targets)

        return loss

class MarginRankingLoss(torch.nn.Module):
    def __init__(self, config):
        super(MarginRankingLoss, self).__init__()
        base = 0.2
        self.ranking = [torch.nn.MarginRankingLoss(margin=base*0.1), torch.nn.MarginRankingLoss(margin=base * 0.5),
                        torch.nn.MarginRankingLoss(margin=base)]
        self.negative_ratio = config.data.negative_ratio


    def forward(self, text_repre, label_repre_positive, label_repre_negative, mask=None):
        # loss inter total : positive sample에 관한 loss
        # loss intra total : negative sample에 관한 loss 

        loss_inter_total, loss_intra_total = 0, 0

        text_score = text_repre.unsqueeze(1).repeat(1, label_repre_positive.size(1), 1)
        loss_inter = (torch.pow(text_score - label_repre_positive, 2)).sum(-1)
        loss_inter = F.relu(loss_inter / text_repre.size(-1))
        loss_inter_total += loss_inter.mean()

        for i in range(self.negative_ratio):
            m = mask[:, i]
            m = m.unsqueeze(-1).repeat(1, 1, label_repre_negative.size(-1))
            label_n_score = torch.masked_select(label_repre_negative, m)
            label_n_score = label_n_score.view(text_repre.size(0), -1, label_repre_negative.size(-1))
            text_score = text_repre.unsqueeze(1).repeat(1, label_n_score.size(1), 1)

            # index 0: parent node
            if i == 0:
                loss_inter_parent = (torch.pow(text_score - label_n_score, 2)).sum(-1)
                loss_inter_parent = F.relu((loss_inter_parent-0.01) / text_repre.size(-1))
                loss_inter_total += loss_inter_parent.mean()
            else:
                # index 1: wrong sibling, index 2: other wrong label
                loss_intra = (torch.pow(text_score - label_n_score, 2)).sum(-1)
                loss_intra = F.relu(loss_intra / text_repre.size(-1))
                loss_gold = loss_inter.view(1, -1)
                loss_cand = loss_intra.view(1, -1)
                ones = torch.ones(loss_gold.size()).to(loss_gold.device)
                loss_intra_total += self.ranking[i](loss_cand, loss_gold, ones)

        return loss_inter_total, loss_intra_total
    
def get_constant_schedule(optimizer, last_epoch=-1):
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)

def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
