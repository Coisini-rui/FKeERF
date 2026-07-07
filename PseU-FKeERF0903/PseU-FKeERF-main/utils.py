# encoding: utf-8

import numpy as np
import time


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


