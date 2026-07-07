# encoding: utf-8

import numpy as np
import time
import random

def cal_label_dic(label_col):
    dic = {}
    for label in label_col:
        if label not in dic:
            dic[label] = 0
        dic[label] += 1
    return dic


def cal_gini(label_column):
    total = len(label_column)
    label_dic = cal_label_dic(label_column)
    imp = 0
    for k1 in label_dic:
        p1 = float(label_dic[k1]) / total
        for k2 in label_dic:
            if k1 == k2: continue
            p2 = float(label_dic[k2]) / total
            imp += p1 * p2
    return imp


def voting(label_dic, b=None):
    if b == None:
        winner_key = list(label_dic.keys())[0]
        for key in label_dic:
            if label_dic[key] > label_dic[winner_key]:
                winner_key = key
            elif label_dic[key] == label_dic[winner_key]:
                winner_key = np.random.choice([key, winner_key], 1)[0]  # return a list with len 1
    else:
        arr = np.array(list(label_dic.items()))
        prob = np.exp(arr[:, 1] * b) / np.exp(arr[:, 1] * b).sum()
        winner_key = np.random.choice(arr[:, 0], size=1, p=prob)[0]

    return winner_key


def max_min_normalization(arr):
    min_ = np.min(arr)
    max_ = np.max(arr)
    if max_ - min_ == 0:
        return np.zeros(np.shape(arr))
    return (arr - min_) / (max_-min_)


def output_time(flag):
    print(flag, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def convert_to_mass(labels):
    unique_labels = sorted(list(set(labels)))

    num_classes = len(unique_labels)

    n = len(labels)

    mass_functions = []

    if n == 0:
        mass_function = np.full(num_classes, 1 / num_classes)
        mass_functions.append(mass_function)
    else:
        for label in labels:
            mass_function = np.zeros(num_classes + 1)
            if label in unique_labels:
                index = unique_labels.index(label)
                mass_function[index] = random.uniform(0.5, 1.0)
                mass_function[1 - index] = random.uniform(0, 0.5)
                mass_function[2] = (mass_function[0] + mass_function[1]) / 2
            mass_functions.append(mass_function)

    mass_functions = np.array(mass_functions)

    sums = np.sum(mass_functions, axis=1, keepdims=True)
    mass_functions_normalized_array = mass_functions / sums

    return mass_functions_normalized_array


import numpy as np
from sklearn.metrics import confusion_matrix

def to_binary_labels(y):
    y = np.asarray(y).astype(int)
    return (y == 1).astype(int)


def safe_classification_metrics(y_true, y_pred, smooth=0.5):
    """
    基于混淆矩阵的分类指标：
    - 启用平滑
    返回：ACC, MCC, SN, SP
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # 映射到 0 / 1
    y_true_bin = to_binary_labels(y_true)
    y_pred_bin = to_binary_labels(y_pred)

    # 强制 0/1 两类
    con_mat = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = con_mat.ravel()

    # 判断是否极端：如果有任何一项为 0，就用平滑后的；否则用原始的
    use_smooth = (tn == 0) or (fp == 0) or (fn == 0) or (tp == 0)

    if use_smooth:
        tn_s = tn + smooth
        fp_s = fp + smooth
        fn_s = fn + smooth
        tp_s = tp + smooth
    else:
        tn_s, fp_s, fn_s, tp_s = float(tn), float(fp), float(fn), float(tp)

    total_s = tn_s + fp_s + fn_s + tp_s

    # ACC
    acc = (tn_s + tp_s) / total_s if total_s > 0 else 0.0

    # SN = TP / (TP + FN)
    sn = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0

    # SP = TN / (TN + FP)
    sp = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0

    # MCC：用平滑后（或原始）的值显式计算
    denom = np.sqrt((tp_s + fp_s) * (tp_s + fn_s) * (tn_s + fp_s) * (tn_s + fn_s))
    if denom > 0:
        mcc = ((tp_s * tn_s) - (fp_s * fn_s)) / denom
    else:
        mcc = 0.0

    # 裁剪到合法区间
    acc = float(min(max(acc, 0.0), 1.0))
    sn = float(min(max(sn, 0.0), 1.0))
    sp = float(min(max(sp, 0.0), 1.0))
    mcc = float(min(max(mcc, -1.0), 1.0))

    return acc, mcc, sn, sp
