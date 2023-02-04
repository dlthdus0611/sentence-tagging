#!/usr/bin/env python
# coding:utf-8

import numpy as np
from sklearn.metrics import accuracy_score


def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def evaluate(epoch_predicts, epoch_labels, label_v2i, label_i2v, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = label_v2i
    id2label = label_i2v

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    acc_score_list = []
    for sample_predict, sample_gold in zip(epoch_predicts, epoch_labels):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        acc_up = len(set(sample_gold).intersection(set(sample_predict_id_list)))
        acc_down = len(set(sample_gold).union(set(sample_predict_id_list)))
        acc_score_list.append(acc_up / acc_down)

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(recall_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0
    acc = sum(acc_score_list) / len(acc_score_list)

    #print("acc: ", sum(acc_score_list) / len(acc_score_list))
    return {'acc': acc,
            'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}

def evaluate_matching(epoch_predicts, epoch_labels, threshold=0.5):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    epoch_predicts = np.array(epoch_predicts)
    epoch_labels = np.array(epoch_labels)
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_predicts = np.array(epoch_predicts).reshape((-1))
    epoch_labels = np.array(epoch_labels).reshape((-1))
    epoch_predicts[epoch_predicts > threshold] = 1
    epoch_predicts[epoch_predicts <= threshold] = 0

    acc = 0
    for i in range(len(epoch_labels)):
        if epoch_labels[i] == epoch_predicts[i]:
            acc += 1
    acc = acc / len(epoch_labels)
    return {'acc': acc}


def evaluate_layer(epoch_predicts, epoch_labels, label_v2i, label_i2v, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
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

    # initialize confusion matrix
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    right_count_list_total = [0 for _ in range(9)]
    gold_count_list_total = [0 for _ in range(9)]
    predicted_count_list_total = [0 for _ in range(9)]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

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

        # print(f'gold_count_list    : {gold_count_list}\nright_count_list   : {right_count_list}\npredict_count_list : {predicted_count_list}\n')
        # print(f'gold_count_list_total    : {gold_count_list_total}\nright_count_list_total   : {right_count_list_total}\npredict_count_list_total : {predicted_count_list_total}\n')
    
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
        # print(f'--------------------- {label} ---------------------')
        # print(f'precision_dict : {precision_dict}\nrecall_dict : {recall_dict}\nfscore_dict : {fscore_dict}')

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
        # print(f'--------------------- {label} ---------------------')
        # print(f'precision_dict : {precision_dict}\nrecall_dict : {recall_dict}\nfscore_dict : {fscore_dict}')

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

    # print('>>>>>>>>>>>>>>>>> micro', precision_micro_layer2 + recall_micro_layer2)
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    # l1-Macro-F1
    fscore_dict_l1 = [v for k, v in fscore_dict.items() if k in rcv1_layer1]
    macro_f1_layer1 = sum(fscore_dict_l1) / len(fscore_dict_l1)

    # l2-Macro-F1
    fscore_dict_l2 = [v for k, v in fscore_dict.items() if k not in rcv1_layer1]
    # print('>>>>>>>>>>>>>>>>> macro', fscore_dict_l2)
    macro_f1_layer2 = sum(fscore_dict_l2) / len(fscore_dict_l2)

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'l1_micro_f1': micro_f1_layer1,
            'l2_micro_f1': micro_f1_layer2,
            'l1_macro_f1': macro_f1_layer1,
            'l2_macro_f1': macro_f1_layer2
            }


def ranking_recall_score(y_true, y_score, k=10):
    # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
    """Recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / n_pos


def mean_recall_k(y_true, y_score, k=10):
    """Mean recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean recall @k : float
    """

    r_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            r_ks.append(ranking_recall_score(y_t, y_s, k=k))

    return np.mean(r_ks)


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def mean_ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    Mean NDCG @k : float
    """

    ndcg_s = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            ndcg_s.append(ndcg_score(y_t, y_s, k=k, gains=gains))

    return np.mean(ndcg_s)


def evaluate_recall_ndcg(epoch_predicts, epoch_labels, label_v2i, label_i2v, threshold=0.5, top_k=None, label_map=None, dataset="eurlex"):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_gold = []
    epoch_predict = []
    epoch_gold_frequent = []
    epoch_predict_frequent = []
    epoch_gold_few = []
    epoch_predict_few = []
    epoch_gold_zero = []
    epoch_predict_zero = []

    frequent_mask = np.array([False] * len(label_map))
    few_mask = np.array([False] * len(label_map))
    zero_mask = np.array([False] * len(label_map))

    with open('./data/'+dataset+'_label_type.txt') as f:
        type_dic = eval(f.readline())
    for l in type_dic['frequent']:
        frequent_mask[label_map[l]] = True
    for l in type_dic['few']:
        few_mask[label_map[l]] = True
    for l in type_dic['zero']:
        zero_mask[label_map[l]] = True

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_labels):
        np_sample_gold = np.zeros((len(sample_predict)), dtype=np.float32)
        for l in sample_gold:
            np_sample_gold[l] = 1
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        epoch_predict.append(np_sample_predict)
        epoch_gold.append(np_sample_gold)

        epoch_gold_frequent.append(np_sample_gold[frequent_mask])
        epoch_gold_few.append(np_sample_gold[few_mask])
        epoch_gold_zero.append(np_sample_gold[zero_mask])

        epoch_predict_frequent.append(np_sample_predict[frequent_mask])
        epoch_predict_few.append(np_sample_predict[few_mask])
        epoch_predict_zero.append(np_sample_predict[zero_mask])

    # epoch_gold = epoch_gold[:3]
    # epoch_predict = epoch_predict[:3]
    recall5 = mean_recall_k(epoch_gold, epoch_predict, top_k)
    ndcg5 = mean_ndcg_score(epoch_gold, epoch_predict, top_k)

    recall5_frequent = mean_recall_k(epoch_gold_frequent, epoch_predict_frequent, top_k)
    ndcg5_frequent = mean_ndcg_score(epoch_gold_frequent, epoch_predict_frequent, top_k)

    recall5_few = mean_recall_k(epoch_gold_few, epoch_predict_few, top_k)
    ndcg5_few = mean_ndcg_score(epoch_gold_few, epoch_predict_few, top_k)

    recall5_zero = mean_recall_k(epoch_gold_zero, epoch_predict_zero, top_k)
    ndcg5_zero = mean_ndcg_score(epoch_gold_zero, epoch_predict_zero, top_k)

    return {'recall': recall5,
            'ndcg': ndcg5,
            'recall_frequent': recall5_frequent,
            'ndcg_frequent': ndcg5_frequent,
            'recall_few': recall5_few,
            'ndcg_few': ndcg5_few,
            'recall_zero': recall5_zero,
            'ndcg_zero': ndcg5_zero}


if __name__ == "__main__":
    epoch_gold = np.array([[1, 1, 0, 0, 0]])
    epoch_predict = np.array([[0.68, 0.23, 0.58, 0, 0]])
    print(mean_recall_k(epoch_gold, epoch_predict, 5))
    print(mean_ndcg_score(epoch_gold, epoch_predict, 5))