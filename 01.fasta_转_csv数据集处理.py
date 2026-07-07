import pandas as pd

def fasta_to_csv(fasta_path, csv_path):
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
    df.to_csv(csv_path, index=False)

# 使用方法：
fasta_to_csv('dataset/H.fasta', 'H.csv')
fasta_to_csv('dataset/M.fasta', 'M.csv')
fasta_to_csv('dataset/S.fasta', 'S.csv')
