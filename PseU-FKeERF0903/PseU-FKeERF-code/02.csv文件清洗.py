import pandas as pd
import numpy as np
from itertools import product


def run(df, input_filename):
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
    kmer_features_df = pd.DataFrame(
        kmer_features.tolist(),
        columns=['kmer_' + kmer for kmer in [''.join(p) for p in product('ACGU', repeat=2)]]
    )

    # 合并Label和特征矩阵
    clean_df = pd.concat([df['Label'], kmer_features_df], axis=1)

    # 根据输入文件名生成输出文件名
    base_name = input_filename.split('.')[0]  # 去掉扩展名
    output_name = f"{base_name}_cleaned.csv"
    clean_df.to_csv(output_name, index=False)

    print(f"数据清洗及特征提取完成，已保存到 {output_name}")


if __name__ == '__main__':

    # 定义要处理的文件列表
    input_files = ['H.csv', 'M.csv', 'S.csv']

    # 用循环处理多个文件
    for fname in input_files:
        df = pd.read_csv(fname)
        run(df, fname)