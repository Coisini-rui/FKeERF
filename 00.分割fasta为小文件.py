def extract_first_n_seqs(fasta_file, output_file, max_seqs=8):
    """提取前N个完整序列（而不是前N行）"""
    count = 0
    current_seq_lines = []

    with open(fasta_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith('>'):
                # 如果是新的序列开始
                if count >= max_seqs:
                    break

                if current_seq_lines:
                    # 写入上一个完整的序列
                    outfile.writelines(current_seq_lines)
                    count += 1

                # 开始新的序列
                current_seq_lines = [line]
            else:
                current_seq_lines.append(line)

        # 写入最后一个序列（如果是完整的）
        if current_seq_lines and count < max_seqs:
            outfile.writelines(current_seq_lines)

    print(f"提取了 {count} 个完整序列到 {output_file}")


# 使用 - 提取前8个完整序列而不是30行
extract_first_n_seqs('dataset/H.fasta', 'demo_test_datasets/H_demo.fasta', 80)