from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import decision_tree_imperfect
import ibelief
import numpy as np
import math
from multiprocessing import Pool

from utils import convert_to_mass


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
            raise NotFittedError("The classifier hasn't been fitted yet")

        try:
            # 尝试获取predict的返回值
            result = self.predict(X, return_bba=True)

            # 如果是元组，说明predict已经修复
            if isinstance(result, tuple) and len(result) == 2:
                _, bbas = result
                predictions = ibelief.decisionDST(bbas.T, 4, return_prob=True)
                return predictions
            else:
                # 否则使用默认概率
                n_samples = X.shape[0]
                return np.ones((n_samples, 2)) * 0.5

        except Exception as e:
            print(f"predict_proba出错，使用默认概率: {e}")
            n_samples = X.shape[0]
            return np.ones((n_samples, 2)) * 0.5


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


class DualChannelERF(BaseEstimator, ClassifierMixin):
    """双通道证据随机森林"""

    def __init__(self, n_estimators=100, min_samples_leaf=4,
                 sequence_weight=0.6, structure_weight=0.4):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.sequence_weight = sequence_weight  # α
        self.structure_weight = structure_weight  # β

        # 初始化两个通道的随机森林
        self.sequence_erf = ERF(n_estimators=n_estimators // 2,
                                min_samples_leaf=min_samples_leaf)
        self.structure_erf = ERF(n_estimators=n_estimators // 2,
                                 min_samples_leaf=min_samples_leaf)

    def fit(self, X_sequence, X_structure, y):
        """分别训练序列通道和结构通道"""
        # 转换质量函数
        y_belief = convert_to_mass(y)

        # 训练序列通道
        self.sequence_erf.fit(X_sequence, y_belief)

        # 训练结构通道
        self.structure_erf.fit(X_structure, y_belief)

        return self

    def predict(self, X_sequence, X_structure):
        """双通道加权融合预测"""
        # 获取两个通道的证据质量
        seq_predictions = self.sequence_erf.predict_proba(X_sequence)
        struct_predictions = self.structure_erf.predict_proba(X_structure)

        # 证据质量加权融合
        fused_predictions = self._weighted_fusion(seq_predictions,
                                                  struct_predictions)

        return fused_predictions

    def _weighted_fusion(self, m_seq, m_struct):
        """证据质量静态加权融合"""
        # m_fused = α * m_seq + β * m_struct
        fused = (self.sequence_weight * m_seq +
                 self.structure_weight * m_struct)
        return fused