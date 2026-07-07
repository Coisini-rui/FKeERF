import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def load_data(dataset_dir):
    """
    从给定的目录加载FASTA文件，返回RNA序列列表。
    dataset_dir: 数据集目录路径
    """
    dataset_paths = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith('.fasta')]
    sequences = []

    for file_path in dataset_paths:
        with open(file_path, 'r') as file:
            sequence = ""
            for line in file:
                line = line.strip()
                if line.startswith('>'):  # 处理FASTA文件中的描述行
                    if sequence:
                        sequences.append(sequence)  # 将上一个序列添加到列表
                    sequence = ""  # 重置当前序列
                else:
                    sequence += line  # 累加序列内容
            if sequence:  # 添加最后一个序列
                sequences.append(sequence)

    return sequences


def one_hot_encode(sequences, maxlen=None):    # maxlen可以自定义修改
    """
    对RNA序列进行独热编码，并填充序列长度（PyTorch版本）。
    sequences: 字符串列表，RNA序列。
    maxlen: 序列的最大长度，若未指定则填充到最长序列长度。
    返回一个Tensor，形状为 (num_sequences, seq_length, 4)。
    """
    nucleotide_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

    encoded_sequences = []

    # 对每个序列进行独热编码
    for seq in sequences:
        encoded_seq = torch.zeros((len(seq), 4))  # 初始化为全0矩阵
        for i, nucleotide in enumerate(seq):
            encoded_seq[i, nucleotide_dict.get(nucleotide, -1)] = 1  # 对应位置赋值为1
        encoded_sequences.append(encoded_seq)

    # 填充序列，确保所有序列的长度一致
    if maxlen is None:
        maxlen = max([seq.shape[0] for seq in encoded_sequences])  # 取最长序列长度

    # 对每个序列进行填充，确保它们的长度一致
    padded_sequences = []
    for seq in encoded_sequences:
        pad_size = maxlen - seq.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, 4))  # 填充部分，填充0
            padded_seq = torch.cat((seq, pad), dim=0)  # 将填充部分加到序列后
        else:
            padded_seq = seq  # 不需要填充，序列本身已经够长
        padded_sequences.append(padded_seq)

    # 转换为Tensor，返回
    return torch.stack(padded_sequences)


def process_data(dataset_dir):
    """
    处理数据：加载、编码。
    dataset_dir: 数据集目录路径
    """
    sequences = load_data(dataset_dir)  # 读取RNA序列
    encoded_sequences = one_hot_encode(sequences)  # 独热编码
    return encoded_sequences


def split_data(X, y, test_size=0.2):
    """
    数据拆分函数，将数据集划分为训练集和测试集。
    X: 输入特征
    y: 标签
    test_size: 测试集比例
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


# 测试脚本部分

def test_data_processing():
    # 假设你已经有了一个包含FASTA文件的目录
    dataset_dir = 'dataset'

    # 1. 测试load_data
    sequences = load_data(dataset_dir)
    assert len(sequences) > 0, "Failed to load sequences!"
    print(f"Loaded {len(sequences)} sequences.")

    # 2. 测试one_hot_encode
    encoded_sequences = one_hot_encode(sequences)
    assert encoded_sequences.shape[0] == len(sequences), "Mismatch in number of sequences!"
    print(f"Encoded {len(sequences)} sequences, each of shape {encoded_sequences.shape[1]}.")

    # 3. 测试process_data
    processed_data = process_data(dataset_dir)
    assert processed_data.shape[0] == len(sequences), "Mismatch in processed data!"
    print("Data processed successfully.")


if __name__ == "__main__":
    test_data_processing()


