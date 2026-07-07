import pandas as pd
import numpy as np
from itertools import product


def fasta_to_kmer_features(fasta_path, csv_path):
    """
    直接从FASTA文件生成k-mer特征CSV文件
    一步到位，不需要中间文件

    Args:
        fasta_path: 输入FASTA文件路径
        csv_path: 输出k-mer特征CSV文件路径
    """
    print(f"正在处理: {fasta_path}")

    # 1. 读取FASTA文件并转换为DataFrame
    with open(fasta_path, 'r') as f:
        lines = f.readlines()

    data = []
    seq_id = ""
    seq = ""

    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if seq_id:
                data.append([seq_id, seq])
            seq_id = line[1:]  # 去掉 >
            seq = ""
        else:
            seq += line
    # 添加最后一条序列
    if seq_id:
        data.append([seq_id, seq])

    df = pd.DataFrame(data, columns=['Sequence_ID', 'Sequence'])
    print(f"  读取了 {len(df)} 条序列")

    # 2. 从Sequence_ID中提取标签
    # 格式假设为: "P1|1|training"，中间的1是标签
    def extract_label(seq_id):
        try:
            parts = seq_id.split('|')
            if len(parts) > 1:
                label = int(parts[1])
                return 1 if label == 1 else -1
        except:
            pass
        # 默认返回正样本
        return 1

    df['Label'] = df['Sequence_ID'].apply(extract_label)
    print(f"  标签统计: 正样本(+1): {sum(df['Label'] == 1)}, 负样本(-1): {sum(df['Label'] == -1)}")

    # 3. 定义二核苷酸 (k-mer) 特征函数
    def get_kmer_freq(seq, k=2):
        kmers = [''.join(p) for p in product('ACGU', repeat=k)]
        kmer_dict = dict.fromkeys(kmers, 0)

        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1

        total = sum(kmer_dict.values())
        if total > 0:
            for key in kmer_dict:
                kmer_dict[key] /= total

        return list(kmer_dict.values())

    # 4. 对每个序列计算特征矩阵
    print("  正在计算k-mer特征...")
    kmer_features = df['Sequence'].apply(lambda x: get_kmer_freq(x))

    # 生成列名
    kmer_columns = ['kmer_' + ''.join(p) for p in product('ACGU', repeat=2)]
    kmer_features_df = pd.DataFrame(kmer_features.tolist(), columns=kmer_columns)

    print(f"  生成了 {len(kmer_columns)} 个k-mer特征")

    # 5. 合并Label和特征矩阵
    clean_df = pd.concat([df['Label'], kmer_features_df], axis=1)

    # 6. 保存到CSV
    clean_df.to_csv(csv_path, index=False)
    print(f"  特征提取完成，已保存到: {csv_path}")
    print(f"  最终数据形状: {clean_df.shape}")

    return clean_df


if __name__ == '__main__':
    # 处理演示数据集
    fasta_to_kmer_features(
        'demo_test_datasets/H_demo.fasta',
        'demo_test_datasets/H_demo.csv'
    )

    # 如果需要处理其他文件，可以这样：
    # fasta_to_kmer_features('dataset/H.fasta', 'H_cleaned.csv')
    # fasta_to_kmer_features('dataset/M.fasta', 'M_cleaned.csv')
    # fasta_to_kmer_features('dataset/S.fasta', 'S_cleaned.csv')