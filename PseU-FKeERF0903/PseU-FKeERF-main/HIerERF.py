# erf.py 新增类
class HierarchicalERF:
    def __init__(self):
        self.local_trees = ERF(n_estimators=50)  # 局部特征树
        self.global_trees = ERF(n_estimators=100) # 全局特征树

    def fit(self, X, y):
        # 分离特征
        X_local = X[:, :16]    # 前16维为k-mer特征
        X_global = X[:, 16:]   # 后24维为结构特征
        
        self.local_trees.fit(X_local, y)
        self.global_trees.fit(X_global, y)

    def predict(self, X):
        local_mass = self.local_trees.predict_mass(X[:, :16])
        global_mass = self.global_trees.predict_mass(X[:, 16:])
        return self._fuse(local_mass, global_mass)  # 层次融合
