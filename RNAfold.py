import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt


class SimpleRNAFold:
    """简化的RNAfold模拟器"""

    def __init__(self):
        # 简化的热力学参数（仅用于演示，非真实值）
        self.energy_params = {
            'AU': -1.0, 'UA': -1.0,
            'GC': -2.0, 'CG': -2.0,
            'GU': -0.5, 'UG': -0.5,
            'mismatch': 0.0  # 不匹配的能量
        }

        # 碱基配对规则
        self.pairing_rules = {
            'A': ['U'], 'U': ['A', 'G'],
            'G': ['C', 'U'], 'C': ['G']
        }

    def can_pair(self, base1, base2):
        """检查两个碱基是否能配对"""
        return base2 in self.pairing_rules.get(base1, [])

    def get_pair_energy(self, base1, base2):
        """获取配对能量"""
        pair = base1 + base2
        return self.energy_params.get(pair, self.energy_params['mismatch'])

    def predict_structure(self, sequence):
        """
        简化的Nussinov算法预测RNA二级结构
        返回：(结构, 能量)
        """
        n = len(sequence)

        # 初始化DP表
        dp = np.zeros((n, n), dtype=float)
        traceback = np.zeros((n, n), dtype=int)

        # 填充DP表
        for length in range(1, n):
            for i in range(n - length):
                j = i + length

                # 情况1: i和j配对
                max_score = -float('inf')
                if self.can_pair(sequence[i], sequence[j]):
                    pair_score = self.get_pair_energy(sequence[i], sequence[j])
                    if i + 1 < j - 1:
                        pair_score += dp[i + 1, j - 1]
                    max_score = pair_score
                    traceback[i, j] = 1  # 配对

                # 情况2: i不配对
                score1 = dp[i + 1, j] if i + 1 < n else 0
                if score1 > max_score:
                    max_score = score1
                    traceback[i, j] = 2  # i不配对

                # 情况3: j不配对
                score2 = dp[i, j - 1] if j - 1 >= 0 else 0
                if score2 > max_score:
                    max_score = score2
                    traceback[i, j] = 3  # j不配对

                # 情况4: 分岔
                for k in range(i + 1, j):
                    score3 = dp[i, k] + dp[k + 1, j]
                    if score3 > max_score:
                        max_score = score3
                        traceback[i, j] = k + 4  # k是分界点

                dp[i, j] = max_score

        # 回溯得到结构
        structure = ['.'] * n
        self._traceback(sequence, 0, n - 1, traceback, structure)

        return ''.join(structure), dp[0, n - 1]

    def _traceback(self, seq, i, j, traceback, structure):
        """回溯得到结构"""
        if i >= j:
            return

        decision = traceback[i, j]

        if decision == 1:  # i和j配对
            structure[i] = '('
            structure[j] = ')'
            self._traceback(seq, i + 1, j - 1, traceback, structure)
        elif decision == 2:  # i不配对
            self._traceback(seq, i + 1, j, traceback, structure)
        elif decision == 3:  # j不配对
            self._traceback(seq, i, j - 1, traceback, structure)
        else:  # 分岔
            k = decision - 4
            self._traceback(seq, i, k, traceback, structure)
            self._traceback(seq, k + 1, j, traceback, structure)

    def calculate_base_pair_probability(self, sequence, num_samples=1000):
        """
        蒙特卡洛方法估计碱基配对概率
        返回：配对概率矩阵
        """
        n = len(sequence)
        prob_matrix = np.zeros((n, n))

        for _ in range(num_samples):
            # 随机生成一个结构（简化）
            structure = self._random_structure(sequence)

            # 记录配对
            stack = []
            for i, sym in enumerate(structure):
                if sym == '(':
                    stack.append(i)
                elif sym == ')':
                    if stack:
                        j = stack.pop()
                        prob_matrix[i, j] += 1
                        prob_matrix[j, i] += 1

        # 归一化
        if num_samples > 0:
            prob_matrix /= num_samples

        return prob_matrix

    def _random_structure(self, sequence):
        """随机生成一个可能的二级结构（简化）"""
        n = len(sequence)
        structure = ['.'] * n

        # 随机决定哪些位置可能配对
        for i in range(n):
            for j in range(i + 4, min(n, i + 20)):  # 限制环的大小
                if np.random.random() < 0.1 and self.can_pair(sequence[i], sequence[j]):
                    if structure[i] == '.' and structure[j] == '.':
                        structure[i] = '('
                        structure[j] = ')'

        return ''.join(structure)

    def calculate_structural_entropy(self, prob_matrix):
        """计算结构熵"""
        n = prob_matrix.shape[0]
        entropy = np.zeros(n)

        for i in range(n):
            # 位置i的配对概率分布
            probs = []

            # 不配对的概率
            unpaired_prob = 1.0 - np.sum(prob_matrix[i, :])
            if unpaired_prob > 0:
                probs.append(unpaired_prob)

            # 与每个位置配对的概率
            for j in range(n):
                if prob_matrix[i, j] > 0:
                    probs.append(prob_matrix[i, j])

            # 计算香农熵
            entropy[i] = -np.sum([p * math.log(p + 1e-10) for p in probs])

        return entropy

    def extract_all_features(self, sequence):
        """
        提取三个维度的特征
        返回：特征字典
        """
        # 1. 能量维度
        structure, mfe = self.predict_structure(sequence)

        # 2. 拓扑维度
        prob_matrix = self.calculate_base_pair_probability(sequence, num_samples=500)

        # 3. 动态涨落维度
        entropy = self.calculate_structural_entropy(prob_matrix)

        features = {
            'sequence': sequence,
            'structure': structure,
            'mfe': mfe,  # 最小自由能
            'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence),
            'prob_matrix': prob_matrix,
            'positional_entropy': entropy,
            'avg_entropy': np.mean(entropy),
            'max_entropy': np.max(entropy),
            'num_base_pairs': structure.count('(')
        }

        return features

    def visualize_structure(self, sequence, structure=None):
        """可视化RNA结构"""
        if structure is None:
            structure, _ = self.predict_structure(sequence)

        # 创建简单的文本可视化
        print(f"序列: {sequence}")
        print(f"结构: {structure}")

        # 绘制配对概率热图
        prob_matrix = self.calculate_base_pair_probability(sequence, num_samples=1000)

        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.title(f"MFE结构 (能量: {self.predict_structure(sequence)[1]:.2f})")
        self._plot_structure(structure)

        plt.subplot(132)
        plt.title("碱基配对概率热图")
        plt.imshow(prob_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='配对概率')
        plt.xlabel('位置')
        plt.ylabel('位置')

        plt.subplot(133)
        plt.title("位置熵分布")
        entropy = self.calculate_structural_entropy(prob_matrix)
        plt.bar(range(len(entropy)), entropy)
        plt.xlabel('位置')
        plt.ylabel('熵')
        plt.axhline(y=np.mean(entropy), color='r', linestyle='--', label=f'平均熵: {np.mean(entropy):.3f}')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def _plot_structure(self, structure):
        """绘制结构图"""
        x = list(range(len(structure)))
        y = [0] * len(structure)

        # 绘制碱基
        for i, sym in enumerate(structure):
            color = 'blue' if sym == '(' else 'red' if sym == ')' else 'gray'
            plt.plot(i, 0, 'o', color=color, markersize=10)
            plt.text(i, -0.1, sym, ha='center', va='center', fontsize=12, fontweight='bold')

        # 绘制配对线
        stack = []
        for i, sym in enumerate(structure):
            if sym == '(':
                stack.append(i)
            elif sym == ')':
                if stack:
                    j = stack.pop()
                    # 绘制弧线
                    x_arc = np.linspace(j, i, 100)
                    y_arc = 0.2 * np.sin(np.pi * (x_arc - j) / (i - j))
                    plt.plot(x_arc, y_arc, 'k-', alpha=0.5)

        plt.xlim(-1, len(structure))
        plt.ylim(-0.5, 0.5)
        plt.axis('off')


# 使用示例
def main():
    # 创建模拟器
    rnafold = SimpleRNAFold()

    # 测试序列
    test_sequences = [
        "GGGAAACCC",  # 简单发夹
        "AUGCUAGCUAGCUAGCU",  # 随机序列
        "GGGAAAUUCCC",  # 发夹结构
    ]

    for seq in test_sequences:
        print(f"\n{'=' * 50}")
        print(f"分析序列: {seq}")
        print(f"{'=' * 50}")

        # 提取所有特征
        features = rnafold.extract_all_features(seq)

        print(f"预测结构: {features['structure']}")
        print(f"最小自由能: {features['mfe']:.2f}")
        print(f"GC含量: {features['gc_content']:.3f}")
        print(f"碱基对数量: {features['num_base_pairs']}")
        print(f"平均位置熵: {features['avg_entropy']:.3f}")

        # 可视化
        rnafold.visualize_structure(seq)

        # 保存特征到文件
        save_features_to_file(features, f"rna_features_{seq[:5]}.txt")


def save_features_to_file(features, filename):
    """保存特征到文件"""
    with open(filename, 'w') as f:
        f.write("# RNA结构特征文件\n")
        f.write(f"序列: {features['sequence']}\n")
        f.write(f"结构: {features['structure']}\n")
        f.write(f"最小自由能: {features['mfe']:.4f}\n")
        f.write(f"GC含量: {features['gc_content']:.4f}\n")
        f.write(f"平均位置熵: {features['avg_entropy']:.4f}\n")
        f.write(f"最大位置熵: {features['max_entropy']:.4f}\n")
        f.write(f"碱基对数量: {features['num_base_pairs']}\n")

        # 保存位置熵向量
        f.write("\n# 位置熵向量\n")
        entropy_str = ','.join([f"{e:.4f}" for e in features['positional_entropy']])
        f.write(f"位置熵: [{entropy_str}]\n")

    print(f"特征已保存到: {filename}")


if __name__ == "__main__":
    main()