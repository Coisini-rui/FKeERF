from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import decision_tree_imperfect
import ibelief
import numpy as np
import math
from multiprocessing import Pool


class ERF(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, min_samples_leaf=1, criterion="conflict", rf_max_features="sqrt", n_jobs=4):

        if (criterion not in ["euclidian", "conflict", "jousselme", "uncertainty"]):
            raise ValueError("Wrong selected criterion")

        self._fitted = False
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.rf_max_features = rf_max_features
        self.n_jobs = n_jobs


    def score(self, X, y_true, criterion=3):
        result = self.predict(X, criterion=criterion)
        return result
        pass

    def score_u65(self, X, y_true):
        _, y_pred = self.predict(X, return_bba=True)

        score = 0

        for i in range(X.shape[0]):
            bel = ibelief.mtobel(y_pred[i])
            pl = ibelief.mtopl(y_pred[i])

            if bel[1] >= 0.5:
                if 0 == y_true[i]:
                    score += 1
            elif pl[1] < 0.5:
                if 1 == y_true[i]:
                    score += 1
            else:
                score += (-1.2) * 0.5 ** 2 + 2.2 * 0.5

        score = score / X.shape[0]

        print(score)
        input()
        return score

    def score_ssa(self, X, y_true):
        _, y_pred = self.predict(X, return_bba=True)

        score = 0
        total = 0

        for i in range(X.shape[0]):

            bel = ibelief.mtobel(y_pred[i])
            pl = ibelief.mtopl(y_pred[i])

            if bel[1] >= 0.5:
                if 0 == y_true[i]:
                    score += 1
                total += 1
            elif pl[1] < 0.5:
                if 1 == y_true[i]:
                    score += 1
                total += 1

        score = score / total

        print(score, total)
        input()
        return score

    def score_Jouss(self, X, y_true):
        _, y_pred = self.predict(X, return_bba=True)

        score = 0

        D = ibelief.Dcalculus(y_pred.shape[1])
        for i in range(X.shape[0]):
            true_mass = np.zeros(y_pred.shape[1])
            true_mass[2 ** y_true[i].astype(int)] = 1
            score += ibelief.JousselmeDistance(y_pred[i], true_mass, D=D)

        score = score / X.shape[0]

        return 1 - score

    def get_estimators(self):
        return self.estimators

    def predict_proba(self, X):
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        _, y_pred = self.predict(X, return_bba=True)

        predictions = ibelief.decisionDST(y_pred.T, 4, return_prob=True)

        return predictions

    def predict(self, X, criterion=3, return_bba=False):
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        prediction_result = []
        bbas = []
        for estimator in self.estimators:
            prediction, mass, attribute, attribute_value = estimator.predict(X, return_bba=True)
            prediction_result.append((attribute, attribute_value, prediction[0]))
            bbas.append(mass)
        bbas = np.array(bbas)

        return prediction_result

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        if math.log(y.shape[1] + 1, 2).is_integer():
            y = np.hstack((np.zeros((y.shape[0], 1)), y))
        elif not math.log(y.shape[1], 2).is_integer():
            raise ValueError("y size must be the size of the power set of the frame of discernment")

        self.X_trained = X
        self.y_trained = y
        self.size = self.X_trained.shape[0]
        self._fitted = True
        self.compute_bagging()

        return self

    def compute_bagging(self):
        bootstrap_indices = self._bootstrap()

        self._fit_estimators(bootstrap_indices)


    def _fit_estimators(self, indices):
        self.estimators = np.array([])
        pool = Pool(processes=self.n_jobs)
        jobs_set = []
        for i in range(self.n_estimators):
            tree = decision_tree_imperfect.EDT(min_samples_leaf=self.min_samples_leaf, criterion=self.criterion,
                                               rf_max_features=self.rf_max_features)
            jobs_set.append(pool.apply_async(decision_tree_imperfect.EDT.fit,
                                             (tree, self.X_trained[indices[i]], self.y_trained[indices[i]],)))
        pool.close()
        pool.join()

        for job in jobs_set:
            self.estimators = np.append(self.estimators, job.get())

    def _bootstrap(self):
        bootstrap_indices = []
        for _ in range(self.n_estimators):
            bootstrap_indices.append(np.random.choice(range(self.size), size=self.size))

        return np.array(bootstrap_indices)
