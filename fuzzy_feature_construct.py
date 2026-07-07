import numpy as np
import skfuzzy as fuzz


def gene_ante_fcm(data):
    k = 3
    h = 0.5
    n_examples, n_features = data.shape

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, k, 2, error=1e-6, maxiter=1000, init=None)
    b = np.zeros((k, n_features))

    for i in range(k):
        v1 = np.tile(cntr[i, :], (n_examples, 1))
        U = u[i, :]
        uu = np.tile(U.T, (n_features, 1)).T
        b[i, :] = np.sum((data - v1) ** 2 * uu, axis=0) / np.sum(uu, axis=0) / 1
    b_result = b * h + np.finfo(float).eps

    return cntr, b_result


def calc_x_g(x, v, b):

    n_examples = x.shape[0]
    x_e = np.concatenate((x, np.ones((n_examples, 1))), axis=1)
    k, d = v.shape

    wt = np.zeros((n_examples, k))
    for i in range(k):
        v1 = np.tile(v[i], (n_examples, 1))
        bb = np.tile(b[i], (n_examples, 1))
        wt[:, i] = np.exp(-np.sum(((x - v1) ** 2 / (2*bb)), axis=1))

    wt2 = np.sum(wt, axis=1)

    ss = wt2 == 0
    wt2[ss] = np.finfo(float).eps
    wt = wt / np.tile(wt2.reshape(-1, 1), (1, k))

    x_g = np.empty((n_examples, (d + 1) * k))
    for i in range(k):
        wt1 = wt[:, i]
        wt2 = np.tile(wt1.reshape(-1, 1), (1, d + 1))
        x_g[:, i * (d + 1):(i + 1) * (d + 1)] = x_e * wt2

    return x_g