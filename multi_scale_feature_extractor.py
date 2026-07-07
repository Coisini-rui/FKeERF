import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
from collections import Counter


class MultiScaleFeatureExtractor:
    """多尺度序列特征提取器

    提取短程(k=1-3)、中程(k=4-5)和长程序列特征，
    包括核苷酸组成、物理化学属性和全局统计特征。
    """
    def __init__(self):
        self.feature_names = []
        self.nucleotides = ['A', 'C', 'G', 'T', 'U']

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

    def _kmer_composition(self, seq, k):
        """计算k-mer组成"""

        # 生成所有可能的k-mer
        def generate_kmers(nucleotides, k):
            if k == 1:
                return nucleotides
            else:
                result = []
                for base in nucleotides:
                    for kmer in generate_kmers(nucleotides, k - 1):
                        result.append(base + kmer)
                return result

        # 对RNA序列，考虑A,C,G,U
        nucleotides = self.nucleotides
        all_kmers = generate_kmers(nucleotides, k)

        # 计算每个k-mer的出现频率
        kmer_counts = Counter()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            kmer_counts[kmer] += 1

        # 转换为频率向量
        total_kmers = max(1, len(seq) - k + 1)  # 避免除零
        composition = []
        for kmer in all_kmers:
            freq = kmer_counts.get(kmer, 0) / total_kmers
            composition.append(freq)

        return composition

    def _physicochemical_properties(self, seq):
        """计算物理化学属性（简化版）"""
        # 简化的物理化学属性字典
        prop_dict = {
            'AA': 1.0, 'AC': 0.8, 'AG': 0.6, 'AU': 0.4,
            'CA': 0.8, 'CC': 1.0, 'CG': 0.9, 'CU': 0.7,
            'GA': 0.6, 'GC': 0.9, 'GG': 1.0, 'GU': 0.8,
            'UA': 0.4, 'UC': 0.7, 'UG': 0.8, 'UU': 1.0,
            'AT': 0.4, 'TA': 0.4, 'TT': 1.0, 'TG': 0.6, 'GT': 0.6  # 补充DNA的
        }

        features = []

        # 平均物理化学属性
        total_prop = 0
        count = 0
        for i in range(len(seq) - 1):
            dinucleotide = seq[i:i + 2]
            prop = prop_dict.get(dinucleotide, 0.5)  # 默认值0.5
            total_prop += prop
            count += 1

        if count > 0:
            features.append(total_prop / count)
        else:
            features.append(0.5)

        # 物理化学属性方差
        props = []
        for i in range(len(seq) - 1):
            dinucleotide = seq[i:i + 2]
            prop = prop_dict.get(dinucleotide, 0.5)
            props.append(prop)

        if props:
            features.append(np.var(props))
        else:
            features.append(0)

        return features

    def _medium_range_features(self, seq, window_size=5):
        """提取中程特征"""
        features = []

        if len(seq) < window_size:
            # 如果序列太短，返回默认值
            return [0] * 6

        # 滑动窗口统计
        window_features = []
        for i in range(0, len(seq) - window_size + 1):
            window = seq[i:i + window_size]
            # 计算局部物理化学属性
            window_feature = self._window_physicochemical(window)
            window_features.append(window_feature)

        # 对窗口特征取平均
        if window_features:
            window_array = np.array(window_features)
            mean_features = np.mean(window_array, axis=0)
            std_features = np.std(window_array, axis=0)
            features.extend(mean_features)
            features.extend(std_features)
        else:
            features.extend([0] * 6)

        return features

    def _window_physicochemical(self, window):
        """计算窗口的物理化学属性"""
        features = []

        # GC含量
        gc_count = window.count('G') + window.count('C')
        features.append(gc_count / len(window))

        # AT含量
        at_count = window.count('A') + window.count('T') + window.count('U')
        features.append(at_count / len(window))

        # 信息熵
        features.append(self._sequence_entropy(window))

        return features

    def _long_range_features(self, seq):
        """提取长程特征"""
        features = []

        # GC含量
        gc_content = (seq.count('G') + seq.count('C')) / max(1, len(seq))
        features.append(gc_content)

        # 信息熵
        entropy = self._sequence_entropy(seq)
        features.append(entropy)

        # 自相关函数（简化版，计算前3个滞后的自相关）
        autocorr = self._autocorrelation(seq, max_lag=3)
        features.extend(autocorr)

        # 序列长度归一化
        features.append(len(seq) / 100.0)  # 假设最大长度100，可调整

        return features

    def _sequence_entropy(self, seq):
        """计算序列的信息熵"""
        if not seq:
            return 0

        # 计算每个核苷酸的概率
        nucleotides = ['A', 'C', 'G', 'T', 'U']
        counts = {base: 0 for base in nucleotides}

        for base in seq:
            if base in counts:
                counts[base] += 1

        # 计算熵
        entropy = 0
        total = len(seq)
        for base in nucleotides:
            p = counts[base] / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _autocorrelation(self, seq, max_lag=3):
        """计算序列的自相关函数"""
        # 将序列转换为数值编码
        base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
        numeric_seq = [base_to_num.get(base, 0) for base in seq]

        if len(numeric_seq) < 2:
            return [0] * max_lag

        autocorrs = []
        seq_mean = np.mean(numeric_seq)

        for lag in range(1, max_lag + 1):
            if lag >= len(numeric_seq):
                autocorrs.append(0)
                continue

            numerator = 0
            denominator = 0

            for i in range(len(numeric_seq) - lag):
                numerator += (numeric_seq[i] - seq_mean) * (numeric_seq[i + lag] - seq_mean)
                denominator += (numeric_seq[i] - seq_mean) ** 2

            if denominator != 0:
                autocorrs.append(numerator / denominator)
            else:
                autocorrs.append(0)

        return autocorrs


