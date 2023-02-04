import numpy as np

def _precision_recall_f1(right, predict, total):
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def evaluate_metric(epoch_predicts, epoch_labels, label_v2i, label_i2v, threshold=0.5, top_k=None):
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = label_v2i
    id2label = label_i2v
    
    label_dict = {}

    with open('../data/hierar/hierar.txt', 'r') as f:
        hierar_label = f.readlines()
    for label in hierar_label[1:]:
        label = label.replace('\n','').split('\t')
        label = list(map(lambda x: label2id[x], label))
        label_dict[label[0]] = label[1:]

    epoch_gold = epoch_labels

    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    right_count_list_total = [0 for _ in range(9)]
    gold_count_list_total = [0 for _ in range(9)]
    predicted_count_list_total = [0 for _ in range(9)]

    l1_acc, l2_acc, l1_l2_acc = 0, 0, 0
    
    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        for pred in sample_predict_id_list:   
            if pred == sample_gold[0]:
                l1_acc += 1
            elif pred == sample_gold[1]:
                l2_acc += 1
        
        sample_predict_id_list.sort()
        
        if sample_predict_id_list == sample_gold:
            l1_l2_acc +=1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

        gold_count_list_total[sample_gold[1]-3] += 1
        sample_predict_id_list = np.array(sample_predict_id_list)
        predict_l1 = sample_predict_id_list[sample_predict_id_list<=2] # 대분류
        predict_l2 = sample_predict_id_list[sample_predict_id_list>2] # 소분류
        
        for l1 in predict_l1:
            for l2 in predict_l2:
                if l2 in label_dict[l1]:
                    if l2 == sample_gold[1]:
                        right_count_list_total[sample_gold[1]-3] += 1
                    predicted_count_list_total[l2-3] += 1

    l1_acc /= len(epoch_predicts)
    l2_acc /= len(epoch_predicts)
    l1_l2_acc /= len(epoch_predicts)
    
    precision_dict = dict(); recall_dict = dict(); fscore_dict = dict()
    total_precision_dict = dict(); total_recall_dict = dict(); total_fscore_dict = dict()

    right_total, predict_total, gold_total = 0, 0, 0
    right_total_layer1, predict_total_layer1, gold_total_layer1 = 0, 0, 0
    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0

    rcv1_layer1 = ["연구 목적", "연구 방법", "연구 결과"]

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                              predicted_count_list[i],
                                                                                              gold_count_list[i])

        if label in rcv1_layer1:
            right_total_layer1 += right_count_list[i]
            gold_total_layer1 += gold_count_list[i]
            predict_total_layer1 += predicted_count_list[i]
        else:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]

    for i in range(9):
        total_precision_dict[i], total_recall_dict[i], total_fscore_dict[i] = _precision_recall_f1(right_count_list_total[i],
                                                                                                    predicted_count_list_total[i],
                                                                                                    gold_count_list_total[i])
        right_total += right_count_list_total[i]
        gold_total += gold_count_list_total[i]
        predict_total += predicted_count_list_total[i]
    
    # Macro-F1
    precision_macro = sum([v for _, v in total_precision_dict.items()]) / len(list(total_precision_dict.keys()))
    recall_macro = sum([v for _, v in total_recall_dict.items()]) / len(list(total_recall_dict.keys()))
    macro_f1 = sum([v for _, v in total_fscore_dict.items()]) / len(list(total_fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total if gold_total > 0 else 0.0
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer1 = float(right_total_layer1) / predict_total_layer1 if predict_total_layer1 > 0 else 0.0
    recall_micro_layer1 = float(right_total_layer1) / gold_total_layer1
    micro_f1_layer1 = 2 * precision_micro_layer1 * recall_micro_layer1 / (
            precision_micro_layer1 + recall_micro_layer1) if (
                                                                     precision_micro_layer1 + recall_micro_layer1) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2

    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    # l1-Macro-F1
    fscore_dict_l1 = [v for k, v in fscore_dict.items() if k in rcv1_layer1]
    macro_f1_layer1 = sum(fscore_dict_l1) / len(fscore_dict_l1)

    # l2-Macro-F1
    fscore_dict_l2 = [v for k, v in fscore_dict.items() if k not in rcv1_layer1]
    macro_f1_layer2 = sum(fscore_dict_l2) / len(fscore_dict_l2)

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'accuracy': l1_l2_acc,
            'l1_micro_f1': micro_f1_layer1,
            'l2_micro_f1': micro_f1_layer2,
            'l1_macro_f1': macro_f1_layer1,
            'l2_macro_f1': macro_f1_layer2,
            'l1_accuracy': l1_acc,
            'l2_accuracy': l2_acc
            }

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)