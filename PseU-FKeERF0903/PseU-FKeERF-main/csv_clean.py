import pandas as pd
import numpy as np
from itertools import product

# 读取你的原始CSV文件
df = pd.read_csv('H.csv')

# 提取label标签
df['Label'] = df['Sequence_ID'].apply(lambda x: int(x.split('|')[1]))


# 定义二核苷酸 (k-mer) 特征函数
def get_kmer_freq(seq, k=2):
    kmers = [''.join(p) for p in product('ACGU', repeat=k)]
    kmer_dict = dict.fromkeys(kmers, 0)

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer in kmer_dict:
            kmer_dict[kmer] += 1

    total = sum(kmer_dict.values())
    for key in kmer_dict:
        kmer_dict[key] /= total

    return list(kmer_dict.values())


# 对每个序列计算特征矩阵
kmer_features = df['Sequence'].apply(lambda x: get_kmer_freq(x))
kmer_features_df = pd.DataFrame(kmer_features.tolist(),
                                columns=['kmer_' + kmer for kmer in [''.join(p) for p in product('ACGU', repeat=2)]])

# 合并Label和特征矩阵
clean_df = pd.concat([df['Label'], kmer_features_df], axis=1)

# 保存为新的CSV文件
clean_df.to_csv('H_cleaned.csv', index=False)

print("数据清洗及特征提取完成，已保存到 H_cleaned.csv")
