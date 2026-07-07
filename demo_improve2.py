import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from evidential_random_forest import ERF, DualChannelERF
from sklearn.model_selection import KFold
import fuzzy_feature_construct
from utils import output_time, convert_to_mass, safe_classification_metrics
from multi_scale_feature_extractor import MultiScaleFeatureExtractor
import torch
from data_processing import one_hot_encode
from feature_extraction import FeatureExtractor

ROUND_NUM = 1
FOLD_NUM = 10


def sen(Y_test, Y_pred):
    try:
        con_mat = confusion_matrix(Y_test, Y_pred)
        if con_mat.shape != (2, 2):
            return 0.0
        tp = con_mat[0][0]
        fn = con_mat[0][1]
        denom = tp + fn
        if denom == 0:
            return 0.0
        return tp / denom
    except Exception:
        return 0.0


def spe(Y_test, Y_pred):
    try:
        con_mat = confusion_matrix(Y_test, Y_pred)
        if con_mat.shape != (2, 2):
            return 0.0
        fp = con_mat[1][0]
        tn = con_mat[1][1]
        denom = tn + fp
        if denom == 0:
            return 0.0
        return tn / denom
    except Exception:
        return 0.0



def train_test(train_set, test_set):
    clf = ERF(n_estimators=100, min_samples_leaf=4, criterion="conflict", rf_max_features="sqrt", n_jobs=1)
    y_train_belief = convert_to_mass(train_set[:, 0])
    clf.fit(train_set[:, 1:], y_train_belief)

    An = []
    for row in test_set[:, 1:]:
        An.append(clf.score(row, test_set[:, 0]))
    An = np.array(An)

    predict_result = []
    for row in train_set[:, 1:]:
        predict_result.append(clf.score(row, test_set[:, 0]))
    predict_result = np.array(predict_result)

    M = []
    for i in range(len(An)):
        K = []
        for xi in predict_result:
            count = 0
            for l in range(len(xi)):
                if tuple(xi[l]) == tuple(An[i][l]):
                    count = count + 1
            Ki = (1 / len(An[0])) * count
            K.append(Ki)
        m1 = 0
        m2 = 0
        for j in range(len(predict_result)):
            m1 = m1 + train_set[j, 0] * K[j]
            m2 = m2 + K[j]
        m = m1 / m2
        M.append(m)

    pre_y = [0 for p in range(len(M))]
    for q in range(len(M)):
        if M[q] < 0:
            pre_y[q] = -1
        elif M[q] > 0:
            pre_y[q] = 1

    y_true = test_set[:, 0].astype(int)
    y_pred = np.array(pre_y).astype(int)

    acc, mcc, sn, sp = safe_classification_metrics(y_true, y_pred)
    return acc, mcc, sn, sp


def cross_validation(data, n_folds=10, n_repeats=1):
    """
    执行交叉验证，直接使用 data 中的特征（不再进行模糊变换）
    data: numpy array, 第一列为标签，其余列为特征
    """
    n_samples = len(data)
    n_folds = min(n_folds, n_samples)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"{n_folds}-fold cross validation with {n_repeats} repeats.")

    ACC, MCC, SN, SP = [], [], [], []
    num = 0

    for i in range(n_repeats):
        for train_index, test_index in kf.split(data[:, 1:], data[:, 0]):
            train_set = data[train_index]
            test_set = data[test_index]

            # 直接调用 train_test 进行训练和测试
            acc, mcc, sn, sp = train_test(train_set, test_set)
            ACC.append(acc)
            MCC.append(mcc)
            SN.append(sn)
            SP.append(sp)
            print(f"ROUND[{num+1}] ACC: {acc:.4f}, MCC: {mcc:.4f}, SN: {sn:.4f}, SP: {sp:.4f}")
            print("=" * 50)
            num += 1

    return np.mean(ACC), np.mean(MCC), np.mean(SN), np.mean(SP)


def extract_struct_features_from_csv(struct_df):
    """从结构特征CSV中提取有效特征向量"""
    struct_features_list = []

    for idx, row in struct_df.iterrows():
        features = []
        # 1. 能量维度特征
        features.append(row['mfe'])
        features.append(row['gc_content'])
        features.append(row['num_base_pairs'])
        # 2. 动态维度特征
        features.append(row['avg_entropy'])
        features.append(row['max_entropy'])
        # 3. 位置熵向量统计
        try:
            entropy_vector = np.array(eval(row['positional_entropy'])) if isinstance(row['positional_entropy'],
                                                                                     str) else row['positional_entropy']
            features.append(np.mean(entropy_vector))
            features.append(np.std(entropy_vector))
            features.append(np.median(entropy_vector))
        except:
            features.extend([0.0, 0.0, 0.0])
        struct_features_list.append(features)

    return np.array(struct_features_list)


def extract_sequence_features_from_csv(kmer_df, sequences):
    """提取三种序列特征：基础k-mer + 多尺度序列特征"""
    # 1. 基础k-mer特征（H_demo.csv已有的）
    base_kmer_features = kmer_df.iloc[:, 1:].values  # 16个k-mer特征

    # 2. 多尺度序列特征（用MultiScaleFeatureExtractor新提取）
    extractor = MultiScaleFeatureExtractor()
    multi_scale_features = extractor.extract_sequence_features(sequences)

    # 3. 组合所有序列特征
    all_seq_features = np.hstack([base_kmer_features, multi_scale_features])

    print(f"基础k-mer特征维度: {base_kmer_features.shape}")
    print(f"多尺度序列特征维度: {multi_scale_features.shape}")
    print(f"总序列特征维度: {all_seq_features.shape}")

    return all_seq_features


def triple_channel_cross_validation(base_kmer_features, multi_scale_features, struct_features, labels):
    """三通道交叉验证"""
    print(f"三通道交叉验证:")
    print(f"  基础k-mer特征: {base_kmer_features.shape}")
    print(f"  多尺度序列特征: {multi_scale_features.shape}")
    print(f"  结构特征: {struct_features.shape}")

    n_samples = len(labels)

    # 检查样本数
    if n_samples < 5:
        print(f"⚠️  警告: 样本数({n_samples})过少，建议至少10个样本以获得可靠结果")
        print(f"   当前使用{n_samples}个样本进行演示")

    # 动态调整折数
    if n_samples <= 10:
        print(f"小样本模式: {n_samples} 个样本")

        # 使用留一法或5折交叉验证
        if n_samples <= 5:
            # 留一法交叉验证
            from sklearn.model_selection import LeaveOneOut
            kf = LeaveOneOut()
            n_folds = n_samples
            print(f"使用留一法交叉验证 (LOOCV)")
            use_loo = True
        else:
            # 5折交叉验证
            n_folds = min(5, n_samples)
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            print(f"使用{n_folds}折交叉验证")
            use_loo = False
    else:
        # 正常情况
        n_folds = min(FOLD_NUM, n_samples)  # FOLD_NUM 应该是10
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        use_loo = False

    print(f"{n_folds}-fold cross validation for triple-channel model")

    ACC, MCC, SN, SP = [], [], [], []
    num = 0

    for i in range(ROUND_NUM):  # ROUND_NUM 应该是1
        if use_loo:
            # LeaveOneOut 的 split 方式
            for train_index, test_index in kf.split(base_kmer_features, labels):
                # 分割三种特征
                X_train_base, X_test_base = base_kmer_features[train_index], base_kmer_features[test_index]
                X_train_multi, X_test_multi = multi_scale_features[train_index], multi_scale_features[test_index]
                X_train_struct, X_test_struct = struct_features[train_index], struct_features[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                # 分别构建模糊特征
                v_base, b_base = fuzzy_feature_construct.gene_ante_fcm(X_train_base)
                G_train_base = fuzzy_feature_construct.calc_x_g(X_train_base, v_base, b_base)
                G_test_base = fuzzy_feature_construct.calc_x_g(X_test_base, v_base, b_base)

                v_multi, b_multi = fuzzy_feature_construct.gene_ante_fcm(X_train_multi)
                G_train_multi = fuzzy_feature_construct.calc_x_g(X_train_multi, v_multi, b_multi)
                G_test_multi = fuzzy_feature_construct.calc_x_g(X_test_multi, v_multi, b_multi)

                v_struct, b_struct = fuzzy_feature_construct.gene_ante_fcm(X_train_struct)
                G_train_struct = fuzzy_feature_construct.calc_x_g(X_train_struct, v_struct, b_struct)
                G_test_struct = fuzzy_feature_construct.calc_x_g(X_test_struct, v_struct, b_struct)

                # 合并基础k-mer和多尺度序列特征
                G_train_seq = np.hstack([G_train_base, G_train_multi])
                G_test_seq = np.hstack([G_test_base, G_test_multi])

                # 双通道训练（序列+结构）
                acc, mcc, sn, sp = dual_channel_train_test(
                    G_train_seq, G_train_struct, y_train,
                    G_test_seq, G_test_struct, y_test
                )

                ACC.append(acc)
                MCC.append(mcc)
                SN.append(sn)
                SP.append(sp)
                print(f"ROUND[{num + 1}] ACC: {acc:.4f}, MCC: {mcc:.4f}, SN: {sn:.4f}, SP: {sp:.4f}")
                print("=" * 50)
                num += 1
        else:
            # KFold 的 split 方式
            for train_index, test_index in kf.split(base_kmer_features, labels):
                # 分割三种特征
                X_train_base, X_test_base = base_kmer_features[train_index], base_kmer_features[test_index]
                X_train_multi, X_test_multi = multi_scale_features[train_index], multi_scale_features[test_index]
                X_train_struct, X_test_struct = struct_features[train_index], struct_features[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                # 分别构建模糊特征
                v_base, b_base = fuzzy_feature_construct.gene_ante_fcm(X_train_base)
                G_train_base = fuzzy_feature_construct.calc_x_g(X_train_base, v_base, b_base)
                G_test_base = fuzzy_feature_construct.calc_x_g(X_test_base, v_base, b_base)

                v_multi, b_multi = fuzzy_feature_construct.gene_ante_fcm(X_train_multi)
                G_train_multi = fuzzy_feature_construct.calc_x_g(X_train_multi, v_multi, b_multi)
                G_test_multi = fuzzy_feature_construct.calc_x_g(X_test_multi, v_multi, b_multi)

                v_struct, b_struct = fuzzy_feature_construct.gene_ante_fcm(X_train_struct)
                G_train_struct = fuzzy_feature_construct.calc_x_g(X_train_struct, v_struct, b_struct)
                G_test_struct = fuzzy_feature_construct.calc_x_g(X_test_struct, v_struct, b_struct)

                # 合并基础k-mer和多尺度序列特征
                G_train_seq = np.hstack([G_train_base, G_train_multi])
                G_test_seq = np.hstack([G_test_base, G_test_multi])

                # 双通道训练（序列+结构）
                acc, mcc, sn, sp = dual_channel_train_test(
                    G_train_seq, G_train_struct, y_train,
                    G_test_seq, G_test_struct, y_test
                )

                ACC.append(acc)
                MCC.append(mcc)
                SN.append(sn)
                SP.append(sp)
                print(f"ROUND[{num + 1}] ACC: {acc:.4f}, MCC: {mcc:.4f}, SN: {sn:.4f}, SP: {sp:.4f}")
                print("=" * 50)
                num += 1

    return np.mean(ACC), np.mean(MCC), np.mean(SN), np.mean(SP)


def dual_channel_train_test(train_seq, train_struct, train_labels, test_seq, test_struct, test_labels):
    """双通道训练测试流程"""
    clf = DualChannelERF(
        n_estimators=100,
        min_samples_leaf=4,
        sequence_weight=0.6,
        structure_weight=0.4
    )
    clf.fit(train_seq, train_struct, train_labels)
    predictions = clf.predict(test_seq, test_struct)
    final_predictions = np.argmax(predictions, axis=1)
    acc, mcc, sn, sp = safe_classification_metrics(test_labels, final_predictions)
    return acc, mcc, sn, sp


def load_sequences_from_fasta(fasta_file):
    """从FASTA文件加载序列"""
    sequences = []
    with open(fasta_file, 'r') as f:
        current_seq = ""
        for line in f:
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            sequences.append(current_seq)
    return sequences




def main(kmer_data_file, struct_data_file, sequences_file):
    # 实际上，kmer_data_file 和 struct_data_file 已不再需要，只保留 sequences_file
    # 但为了保持函数签名，可以暂时保留，内部不使用。

    # 1. 加载原始序列
    sequences = load_sequences_from_fasta(sequences_file)
    print(f"加载了 {len(sequences)} 条序列")

    # 2. 加载标签（这里假设标签可以从某个地方获取，例如从 kmer_data_file 的第一列）
    #   如果标签原本在 kmer CSV 中，我们仍需读取它，但只取标签列
    kmer_data = pd.read_csv(kmer_data_file)
    labels = kmer_data.iloc[:, 0].values.astype(np.float32)
    # 确保标签与序列数量一致
    assert len(labels) == len(sequences), "标签数量与序列数量不匹配"

    # ===== 添加：随机生成二分类标签（用于演示）=====
    # 将30%的标签随机改为0，70%保持为1
    """ 之所以要这么做的原因，是因为我们使用的demo文件里面只有1分类，没有2分类。所以要人为创造。就这么简单。后面自己采用真实数据集，有两个分类的话，这一段可以注释掉。"""
    np.random.seed(42)  # 固定种子保证可重复性
    mask = np.random.random(len(labels)) < 0.3  # 30%的概率改为0
    labels[mask] = 0
    print("新标签分布:", np.bincount(labels.astype(int)))
    print(f"正类(1)数量: {np.sum(labels == 1)}, 负类(0)数量: {np.sum(labels == 0)}")
    # ============================================


    # 3. 独热编码
    X_onehot = one_hot_encode(sequences)          # 返回 torch.Tensor, shape (n, maxlen, 4)
    # 转为 (n, 4, maxlen) 格式以适应 FeatureExtractor
    X_onehot = X_onehot.permute(0, 2, 1)           # shape (n, 4, maxlen)

    # 4. 初始化特征提取器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor(
        input_channels=4,
        resnet_channels=32,
        cnn_channels=64,
        lstm_hidden=128,
        gru_hidden=128,
        cnn_kernel_sizes=[1, 3, 5]
    ).to(device)
    extractor.eval()

    # 5. 批量提取深度特征（假设内存允许一次性处理所有样本，否则可分批）
    batch_size = 32
    deep_features = []
    with torch.no_grad():
        for i in range(0, len(X_onehot), batch_size):
            batch = X_onehot[i:i+batch_size].to(device)
            feats = extractor(batch).cpu().numpy()   # shape (batch, feat_dim)
            deep_features.append(feats)
    deep_features = np.vstack(deep_features)          # shape (n_samples, feat_dim)
    print(f"深度特征维度: {deep_features.shape[1]}")

    # 6. 构造最终数据集（标签 + 深度特征）
    data = np.hstack((labels.reshape(-1, 1), deep_features))

    # 7. 执行交叉验证（使用修改后的 cross_validation 函数）
    acc, mcc, sn, sp = cross_validation(data, n_folds=10, n_repeats=1)

    # 8. 输出结果
    print(f"ACC: {acc:.4f}, MCC: {mcc:.4f}, SN: {sn:.4f}, SP: {sp:.4f}")


if __name__ == '__main__':
    # 1. 加载基础k-mer特征和标签
    kmer_data_file = "demo_test_datasets/H_demo.csv"  # 改为文件路径字符串
    # 2. 加载结构特征
    struct_data_file = "demo_test_datasets/H_demo_structure_features.csv"  # 改为文件路径字符串
    # 3. 加载原始序列用于提取多尺度特征
    sequences_file = "demo_test_datasets/H_demo.fasta"  # 改为文件路径字符串

    main(kmer_data_file, struct_data_file, sequences_file)
