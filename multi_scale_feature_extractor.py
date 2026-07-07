import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math


class MultiScaleFeatureExtractor:
    def __init__(self):
        self.feature_names = []

    def extract_sequence_features(self, sequences):
        """提取多尺度序列特征"""
        all_features = []

        for seq in sequences:
            features = []

            # 1. 短程特征 (k=1,2,3)
            features.extend(self._short_range_features(seq))

            # 2. 中程特征 (k=4,5)
            features.extend(self._medium_range_features(seq))

            # 3. 长程/全局特征
            features.extend(self._long_range_features(seq))

            all_features.append(features)

        return np.array(all_features)

    def extract_structure_features(self, sequences):
        """提取多维度结构特征（使用RNAfold模拟）"""
        all_features = []

        for seq in sequences:
            features = []

            # 能量维度 - 模拟最小自由能
            mfe = self._simulate_mfe(seq)
            features.append(mfe)

            # 拓扑维度 - 模拟碱基配对概率
            pairing_prob = self._simulate_pairing_probability(seq)
            features.extend(pairing_prob)

            # 动态涨落维度 - 模拟香农熵
            entropy = self._calculate_structural_entropy(seq)
            features.append(entropy)

            all_features.append(features)

        return np.array(all_features)

    def _short_range_features(self, seq):
        """提取短程特征"""
        features = []
        k_values = [1, 2, 3]

        for k in k_values:
            # 核苷酸组成
            composition = self._kmer_composition(seq, k)
            features.extend(composition)

            # 物理化学属性（简化模拟）
            if k == 2:
                physchem = self._physicochemical_properties(seq)
                features.extend(physchem)

        return features

    def _medium_range_features(self, seq, window_size=5):
        """提取中程特征"""
        features = []

        # 滑动窗口统计
        for i in range(0, len(seq) - window_size + 1):
            window = seq[i:i + window_size]
            # 计算局部物理化学属性
            window_features = self._window_physicochemical(window)
            features.extend(window_features)

        return np.mean(features, axis=0) if features else [0] * 3

    def _long_range_features(self, seq):
        """提取长程特征"""
        features = []

        # GC含量
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        features.append(gc_content)

        # 信息熵
        entropy = self._sequence_entropy(seq)
        features.append(entropy)

        # 自相关函数
        autocorr = self._autocorrelation(seq)
        features.extend(autocorr)

        return features

    def _simulate_mfe(self, seq):
        """模拟RNAfold的最小自由能计算"""
        # 简化的自由能计算（实际应用中应调用RNAfold）
        gc_count = seq.count('G') + seq.count('C')
        return -0.5 * gc_count - 0.3 * len(seq)

    # 其他辅助方法...
    def _kmer_composition(self, seq, k):
        """计算k-mer组成"""
        kmers = {}
        total = len(seq) - k + 1

        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            kmers[kmer] = kmers.get(kmer, 0) + 1

        # 归一化
        return [count / total for count in kmers.values()]

    def _simulate_pairing_probability(self, seq):
        pass

    def _calculate_structural_entropy(self, seq):
        pass

    def _sequence_entropy(self, seq):
        pass

    def _autocorrelation(self, seq):
        pass

    def _window_physicochemical(self, window):
        pass

    def _physicochemical_properties(self, seq):
        pass