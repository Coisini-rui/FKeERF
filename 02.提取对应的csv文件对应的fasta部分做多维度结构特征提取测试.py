import pandas as pd

# 1. 创建一个脚本，从原始FASTA提取对应序列
import pandas as pd

from RNAfold import SimpleRNAFold

# H_demo.csv是从某个FASTA的前2000条序列生成的
fasta_file = "demo_test_datasets/H_demo.fasta"
csv_file = "demo_test_datasets/H_demo.csv"

csv_data = pd.read_csv(csv_file)
n_samples = len(csv_data)

# 从原FASTA提取前n_samples条序列
sequences = []
count = 0
with open(fasta_file, 'r') as f:
    current_seq = ""
    for line in f:
        if line.startswith('>'):
            if current_seq and count < n_samples:
                sequences.append(current_seq)
                count += 1
            current_seq = ""
        else:
            current_seq += line.strip()

    # 最后一条
    if current_seq and count < n_samples:
        sequences.append(current_seq)

# 2. 提取结构特征
extractor = SimpleRNAFold()
struct_features = []

for i, seq in enumerate(sequences):
    if i % 100 == 0:
        print(f"处理进度: {i}/{len(sequences)}")
    features = extractor.extract_all_features(seq)
    struct_features.append(features)

# 3. 保存结构特征
struct_df = pd.DataFrame(struct_features)
struct_df.to_csv("demo_test_datasets/H_demo_structure_features.csv", index=False)

