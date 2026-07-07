import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from evidential_random_forest import ERF
from sklearn.model_selection import KFold
import fuzzy_feature_construct
from utils import output_time, convert_to_mass, safe_classification_metrics

ROUND_NUM = 1
FOLD_NUM = 10



from sklearn.metrics import confusion_matrix

def sen(Y_test, Y_pred):
    try:
        # 尝试正常计算
        con_mat = confusion_matrix(Y_test, Y_pred)
        # 如果不是 2x2，就直接返回一个固定值
        if con_mat.shape != (2, 2):
            return 0.0

        tp = con_mat[0][0]
        fn = con_mat[0][1]
        denom = tp + fn
        if denom == 0:
            return 0.0
        return tp / denom
    except Exception:
        return 0.0


def spe(Y_test, Y_pred):
    try:
        con_mat = confusion_matrix(Y_test, Y_pred)
        if con_mat.shape != (2, 2):
            return 0.0

        fp = con_mat[1][0]
        tn = con_mat[1][1]
        denom = tn + fp
        if denom == 0:
            return 0.0
        return tn / denom
    except Exception:
        return 0.0



def train_test(train_set, test_set):
    clf = ERF(n_estimators=100, min_samples_leaf=4, criterion="conflict", rf_max_features="sqrt", n_jobs=1)
    y_train_belief = convert_to_mass(train_set[:, 0])

    clf.fit(train_set[:, 1:], y_train_belief)

    An = []
    for row in test_set[:, 1:]:
        An.append(clf.score(row, test_set[:, 0]))
    An = np.array(An)

    predict_result = []
    for row in train_set[:, 1:]:
        predict_result.append(clf.score(row, test_set[:, 0]))
    predict_result = np.array(predict_result)

    M = []
    for i in range(len(An)):
        K = []
        for xi in predict_result:
            count = 0
            for l in range(len(xi)):
                if tuple(xi[l]) == tuple(An[i][l]):
                    count = count + 1
            Ki = (1 / len(An[0])) * count
            K.append(Ki)
        m1 = 0
        m2 = 0
        for j in range(len(predict_result)):
            m1 = m1 + train_set[j, 0] * K[j]
            m2 = m2 + K[j]
        m = m1 / m2
        M.append(m)
    # pre_y = [0 for p in range(len(M))]
    # for q in range(len(M)):
    #     if M[q] < 0:
    #         pre_y[q] = -1
    #     elif M[q] > 0:
    #         pre_y[q] = 1
    #
    # return accuracy_score(test_set[:, 0].astype(int), pre_y), \
    #     matthews_corrcoef(test_set[:, 0].astype(int), pre_y), sen(test_set[:, 0].astype(int), pre_y), \
    #     spe(test_set[:, 0].astype(int), pre_y)

    pre_y = [0 for p in range(len(M))]
    for q in range(len(M)):
        if M[q] < 0:
            pre_y[q] = -1
        elif M[q] > 0:
            pre_y[q] = 1

    y_true = test_set[:, 0].astype(int)
    y_pred = np.array(pre_y).astype(int)

    acc, mcc, sn, sp = safe_classification_metrics(y_true, y_pred)
    return acc, mcc, sn, sp


def cross_validation(data):
    print("{0}-fold cross validation with {1} times.".format(str(FOLD_NUM), str(ROUND_NUM)))

    ACC = []
    MCC = []
    SN = []
    SP = []
    num = 0

    for i in range(ROUND_NUM):
        kf = KFold(n_splits=FOLD_NUM, shuffle=True)
        for train_index, test_index in kf.split(X=data[:, 1:], y=data[:, 0], groups=data[:, 0]):
            train_set, test_set = data[train_index], data[test_index]

            v, b = fuzzy_feature_construct.gene_ante_fcm(train_set[:, 1:])
            tr_num = train_set.shape[0]
            data_set = np.concatenate([train_set, test_set], axis=0)

            G_train = fuzzy_feature_construct.calc_x_g(data_set[:, 1:], v, b)
            train_set = np.concatenate((np.array(data_set[:tr_num, 0]).reshape(-1, 1), np.array(G_train[:tr_num, :])), axis=1)
            test_set = np.concatenate((np.array(data_set[tr_num:, 0]).reshape(-1, 1), np.array(G_train[tr_num:, :])), axis=1)

            acc, mcc, sn, sp = train_test(train_set, test_set)
            ACC.append(acc)
            MCC.append(mcc)
            SN.append(sn)
            SP.append(sp)
            print("ROUND[{0}] ACC: {1}".format(str(num + 1), str(acc)))
            print("ROUND[{0}] MCC: {1}".format(str(num + 1), str(mcc)))
            print("ROUND[{0}]  SN: {1}".format(str(num + 1), str(sn)))
            print("ROUND[{0}]  SP: {1}".format(str(num + 1), str(sp)))
            print("============================================")
            num += 1
    return np.mean(ACC), np.mean(MCC), np.mean(SN), np.mean(SP)


import numpy as np
from multi_scale_feature_extractor import MultiScaleFeatureExtractor
from evidential_random_forest import DualChannelERF
import fuzzy_feature_construct


def improved_cross_validation(data, sequences):
    """改进的交叉验证流程"""
    print("使用改进的多尺度特征+双通道证据随机森林")

    # 初始化多尺度特征提取器
    feature_extractor = MultiScaleFeatureExtractor()

    # 提取多尺度特征
    print("提取多尺度序列特征...")
    sequence_features = feature_extractor.extract_sequence_features(sequences)

    print("提取多维度结构特征...")
    structure_features = feature_extractor.extract_structure_features(sequences)

    # 分别构建模糊特征
    print("构建序列模糊特征...")
    v_seq, b_seq = fuzzy_feature_construct.gene_ante_fcm(sequence_features)
    G_seq = fuzzy_feature_construct.calc_x_g(sequence_features, v_seq, b_seq)

    print("构建结构模糊特征...")
    v_struct, b_struct = fuzzy_feature_construct.gene_ante_fcm(structure_features)
    G_struct = fuzzy_feature_construct.calc_x_g(structure_features, v_struct, b_struct)

    # 双通道训练和预测
    acc, mcc, sn, sp = dual_channel_train_test(
        G_seq, G_struct, data[:, 0]
    )

    return acc, mcc, sn, sp


def dual_channel_train_test(train_seq, train_struct, train_labels,
                            test_seq, test_struct, test_labels):
    """双通道训练测试流程"""

    # 初始化双通道ERF
    clf = DualChannelERF(
        n_estimators=100,
        min_samples_leaf=4,
        sequence_weight=0.6,  # α参数
        structure_weight=0.4  # β参数
    )

    # 训练双通道模型
    clf.fit(train_seq, train_struct, train_labels)

    # 预测
    predictions = clf.predict(test_seq, test_struct)

    # 转换为类别标签
    final_predictions = np.argmax(predictions, axis=1)

    # 计算评估指标
    acc, mcc, sn, sp = safe_classification_metrics(test_labels, final_predictions)
    return acc, mcc, sn, sp


if __name__ == '__main__':
    output_time("START")

    # data = pd.read_csv("H_cleaned.csv")
    data = pd.read_csv("H_cleaned2.csv")
    for i in range(len(data.iloc[:, 0])):
        if data.iloc[i, 0] == 0:
            data.iloc[i, 0] = -1

    features = np.array(data.iloc[:, 1:])
    data = np.hstack((np.array(data.iloc[:, 0]).reshape(-1, 1), features))

    acc, mcc, sn, sp = cross_validation(np.array(data))
    print("FINAL ACC: {0}".format(str(acc)[:6]))
    print("FINAL MCC: {0}".format(str(mcc)[:6]))
    print("FINAL  SN: {0}".format(str(sn)[:6]))
    print("FINAL  SP: {0}".format(str(sp)[:6]))

    output_time("END")


