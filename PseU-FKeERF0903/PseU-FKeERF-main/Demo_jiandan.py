import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from evidential_random_forest import ERF
from sklearn.model_selection import KFold
from utils import output_time
import fuzzy_feature_construct

ROUND_NUM = 1
FOLD_NUM = 2

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


def sen(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[0][0]
    fn = con_mat[0][1]
    sen1 = tp / (tp + fn)
    return sen1


def spe(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    fp = con_mat[1][0]
    tn = con_mat[1][1]
    spe1 = tn / (tn + fp)
    return spe1


def train_test(train_set, test_set):
    clf = ERF(n_estimators=2, min_samples_leaf=4, criterion="conflict", rf_max_features="sqrt", n_jobs=4)
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
    pre_y = [0 for p in range(len(M))]
    for q in range(len(M)):
        if M[q] < 0:
            pre_y[q] = -1
        elif M[q] > 0:
            pre_y[q] = 1

    return accuracy_score(test_set[:, 0].astype(int), pre_y), \
        matthews_corrcoef(test_set[:, 0].astype(int), pre_y), sen(test_set[:, 0].astype(int), pre_y), \
        spe(test_set[:, 0].astype(int), pre_y)


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


if __name__ == '__main__':
    output_time("START")

    data = pd.read_csv("H_cleaned.csv")
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
