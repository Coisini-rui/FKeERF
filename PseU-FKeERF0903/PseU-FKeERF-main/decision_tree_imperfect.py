from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
import numpy as np
import math
import ibelief


class EDT(BaseEstimator, ClassifierMixin):

    def __init__(self, min_samples_leaf=1, criterion="conflict", lbda=0.5, rf_max_features="None"):
        if (criterion not in ["euclidian", "conflict", "jousselme", "uncertainty"]):
            raise ValueError("Wrong selected criterion")

        self._fitted = False

        self.root_node = TreeNode()
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.lbda = lbda
        self.rf_max_features = rf_max_features

    def score(self, X, y_true, criterion=3):
        y_pred = self.predict(X, criterion=criterion)
        return accuracy_score(y_true, y_pred)

    def print_tree(self):
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        self.root_node.print_tree()

    def max_depth(self):
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        return (self.root_node.max_depth())

    def mean_samples_leafs(self):
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        return (np.mean(np.array(self.root_node.mean_samples_leafs())))

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            if X.shape[0] * (self.nb_classes + 1) == y.shape[0]:
                y = np.reshape(y, (-1, self.nb_classes + 1))
            else:
                raise ValueError("X and y must have the same number of rows")

        if math.log(y.shape[1] + 1, 2).is_integer():
            y = np.hstack((np.zeros((y.shape[0], 1)), y))
        elif not math.log(y.shape[1], 2).is_integer():
            raise ValueError("y size must be the size of the power set of the frame of discernment")

        self.X_trained = X
        self.y_trained = y

        if (self.criterion == "conflict"):
            self.d_matrix = ibelief.Dcalculus(self.y_trained[0].size)
            self.distances = self._compute_inclusion_distances()

        elif (self.criterion == "jousselme"):
            self.d_matrix = ibelief.Dcalculus(self.y_trained[0].size)
            self.distances = self._compute_jousselme_distances()

        elif (self.criterion == "euclidian"):
            self.distances = self._compute_euclidian_distances()

        elif (self.criterion == "uncertainty"):
            self.pign_prob, self.elements_size = self._compute_prignistic_prob()

        self.size = self.X_trained.shape[0]

        self.root_node = TreeNode()
        self._build_tree(np.array(range(self.size)), self.root_node)

        self._fitted = True

        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        result = np.zeros((X.shape[0], self.y_trained.shape[1]))
        for x in range(X.shape[0]):
            result[x] = self._predict(X[x], self.root_node)

        predictions = ibelief.decisionDST(result.T, 4, return_prob=True)

        return predictions

    def predict(self, X, criterion=3, return_bba=False):
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        attribute, attribute_value, result = self._predict(X, self.root_node)

        if criterion == 1:
            predictions = ibelief.decisionDST(result.T, 1)
        elif criterion == 2:
            predictions = ibelief.decisionDST(result.T, 2)
        elif criterion == 3:
            predictions = ibelief.decisionDST(result.T, 4)
        else:
            raise ValueError("Unknown decision criterion")

        if return_bba:
            return predictions, result, attribute, attribute_value
        else:
            return predictions, attribute, attribute_value

    def _build_tree(self, indices, root_node):
        A, threshold = self._best_gain(indices)

        if A != None:
            if threshold == None:

                for v in np.unique(self.X_trained[indices, A]):
                    index = np.where(self.X_trained[indices, A] == v)[0]
                    node = TreeNode(attribute=A, attribute_value=v)
                    self._build_tree(index, node)
                    root_node.leafs.append(node)

            else:
                index = indices[np.where(self.X_trained[indices, A].astype(float) < threshold)[0]]
                node = TreeNode(attribute=A, attribute_value=threshold, continuous_attribute=1)
                self._build_tree(index, node)
                root_node.leafs.append(node)

                index = indices[np.where(self.X_trained[indices, A].astype(float) >= threshold)[0]]
                node = TreeNode(attribute=A, attribute_value=threshold, continuous_attribute=2)
                self._build_tree(index, node)
                root_node.leafs.append(node)
        else:
            root_node.mass = ibelief.DST(self.y_trained[indices].T, 12).flatten()
            root_node.number_leaf = self.y_trained[indices].shape[0]

    def _best_gain(self, indices):
        info_root = self._compute_info(indices)

        if info_root == 0:
            return None, None

        gains = np.zeros(self.X_trained.shape[1])
        thresholds = []

        selected_features = []
        if self.rf_max_features == "sqrt":
            selected_features = np.random.choice(range(self.X_trained.shape[1]),
                                                 int(math.sqrt(self.X_trained.shape[1])), replace=False)
        else:
            selected_features = range(self.X_trained.shape[1])

        for A in selected_features:
            sum = 0
            threshold = None
            flag_float = True

            try:
                float(self.X_trained[0, A])
            except ValueError:
                flag_float = False

            if flag_float == False:
                for v in np.unique(self.X_trained[indices, A]):
                    node = indices[np.where(self.X_trained[indices, A] == v)[0]]
                    sum += (node.shape[0] / indices.shape[0]) * self._compute_info(indices)

                    if (node.shape[0] < self.min_samples_leaf):
                        sum = info_root
                        break

            # Numerical values
            else:
                # Find best split
                threshold, sum = self._find_treshoold(indices, info_root, A)

            thresholds.append(threshold)

            # Calculate gain
            gain = info_root - sum
            if (gain) > 0:
                gains[A] = gain
            else:
                gains[A] = 0

        if np.max(gains) == 0:
            return None, None

        threshold_arg, = np.where(selected_features == np.argmax(gains))
        return np.argmax(gains), thresholds[threshold_arg[0]]

    def _compute_info(self, indices):
        if indices.shape[0] == 0 or indices.shape[0] == 1:
            return 0

        if self.criterion == 'conflict' or self.criterion == 'jousselme' or self.criterion == "euclidian":
            info = self._compute_distance(indices)
        if self.criterion == 'uncertainty':
            info = self._compute_uncertainty(indices)

        return info

    def _compute_distance(self, indices):
        divisor = indices.shape[0] ** 2 - indices.shape[0]

        mean_distance = np.sum(self.distances[indices][:, indices]) / divisor

        return mean_distance

    def _compute_inclusion_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                d_inc = self._compute_inclusion_degree(self.y_trained[i], self.y_trained[j])
                distances[i, j] = (1 - d_inc) * math.sqrt(
                    np.dot(np.dot(self.y_trained[i] - self.y_trained[j], self.d_matrix),
                           self.y_trained[i] - self.y_trained[j]) / 2.0)

        return distances

    def _compute_jousselme_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                distances[i, j] = math.sqrt(np.dot(np.dot(self.y_trained[i] - self.y_trained[j], self.d_matrix),
                                                   self.y_trained[i] - self.y_trained[j]) / 2.0)

        return distances

    def _compute_euclidian_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                distances[i, j] = math.dist(self.y_trained[i], self.y_trained[j])

        return distances

    def _compute_prignistic_prob(self):
        size = self.y_trained.shape[0]

        pign_prob = np.zeros((size, self.y_trained.shape[1]))
        elemets_size = np.zeros(self.y_trained.shape[1])

        for k in range(size):
            betp_atoms = ibelief.decisionDST(self.y_trained[k].T, 4, return_prob=True)[0]
            for i in range(1, self.y_trained.shape[1]):
                for j in range(betp_atoms.shape[0]):
                    if ((2 ** j) & i) == (2 ** j):
                        pign_prob[k][i] += betp_atoms[j]

        for i in range(1, self.y_trained.shape[1]):
            elemets_size[i] = math.log2(bin(i).count("1"))

        return pign_prob, elemets_size

    def _compute_inclusion_degree(self, m1, m2):
        m1 = m1[:-1]
        m2 = m2[:-1]
        n1 = np.where(m1 > 0)[0]
        n2 = np.where(m2 > 0)[0]

        if n1.shape[0] == 0 or n2.shape[0] == 0:
            return 1

        d_inc_l = 0
        d_inc_r = 0

        for X1 in n1:
            for X2 in n2:
                if X1 & X2 == X1:
                    d_inc_l += 1
                if X1 & X2 == X2:
                    d_inc_r += 1

        return (1 / (n1.shape[0] * n2.shape[0])) * max(d_inc_r, d_inc_l)

    def _compute_uncertainty(self, indices):

        mass = np.mean(self.y_trained[indices], axis=0)
        lbda = self.lbda

        betp_2 = np.mean(self.pign_prob[indices], axis=0)
        betp_2[betp_2 == 0] = 0.001

        n_mass = mass * self.elements_size
        d_mass = -1 * (mass * np.log2(betp_2))

        return ((1 - lbda) * np.sum(n_mass)) + (lbda * np.sum(d_mass))

    def _find_treshoold(self, indices, info_root, A):
        values = np.sort(np.unique(self.X_trained[indices, A]).astype(float))

        thresholds = []
        for i in range(values.shape[0] - 1):
            thresholds.append((values[i] + values[i + 1]) / 2)

        if len(thresholds) == 0:
            return values[0], info_root

        infos = np.zeros(len(thresholds))

        for v in range(len(thresholds)):

            left_node = indices[np.where(self.X_trained[indices, A].astype(float) < thresholds[v])[0]]
            info = (left_node.shape[0] / indices.shape[0]) * self._compute_info(left_node)

            right_node = indices[np.where(self.X_trained[indices, A].astype(float) >= thresholds[v])[0]]
            info += (right_node.shape[0] / indices.shape[0]) * self._compute_info(right_node)

            if (left_node.shape[0] < self.min_samples_leaf or right_node.shape[0] < self.min_samples_leaf):
                info = info_root

            infos[v] = info

        return thresholds[np.argmin(infos)], infos[np.argmin(infos)]

    def _predict(self, X, root_node):
        if type(root_node.mass) is np.ndarray:
            return root_node.attribute, root_node.attribute_value, root_node.mass

        for v in root_node.leafs:
            if v.continuous_attribute == 0 and X[v.attribute] == v.attribute_value:
                return self._predict(X, v)
            elif v.continuous_attribute == 1 and X[v.attribute].astype(float) < v.attribute_value:
                return self._predict(X, v)
            elif v.continuous_attribute == 2 and X[v.attribute].astype(float) >= v.attribute_value:
                return self._predict(X, v)

        print("Classification Error, Tree not complete.")
        return None


class TreeNode():
    def __init__(self, mass=None, attribute=None, attribute_value=0, continuous_attribute=0, number_leaf=0):

        self.leafs = []

        self.mass = mass
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.continuous_attribute = continuous_attribute
        self.number_leaf = number_leaf

    def max_depth(self, depth=1):
        maximum_depth = []
        for i in self.leafs:
            maximum_depth.append(i.max_depth(depth=depth + 1))

        if len(self.leafs) == 0:
            return depth

        return np.max(np.array(maximum_depth))

    def mean_samples_leafs(self):
        samples = []

        for i in self.leafs:
            childs = i.mean_samples_leafs()

            if isinstance(childs, int):
                samples.append(childs)
            else:
                for j in childs:
                    samples.append(j)

        if len(self.leafs) == 0:
            return self.number_leaf

        return samples

    def print_tree(self, depth=1):
        for i in self.leafs:
            if i.continuous_attribute == 0:
                print('|', '---' * depth, "Attribut", i.attribute, " : ", i.attribute_value)
            elif i.continuous_attribute == 1:
                print('|', '---' * depth, "Attribut", i.attribute, "<", i.attribute_value)
            elif i.continuous_attribute == 2:
                print('|', '---' * depth, "Attribut", i.attribute, ">=", i.attribute_value)
            i.print_tree(depth + 1)

        if len(self.leafs) == 0:
            print('    ' * depth, "N:", self.number_leaf, ", Mass : ", np.around(self.mass, decimals=2))

    def add_leaf(self, node):
        self.leafs.append(node)
