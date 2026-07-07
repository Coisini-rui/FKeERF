import torch
import torch.nn as nn
import torch.optim as optim
from feature_extraction import FeatureExtractor  # 导入刚才的特征提取模块

class DeepRNAac4C(nn.Module):
    def __init__(self, input_channels=4, resnet_channels=32, cnn_channels=64,
                 lstm_hidden=128, gru_hidden=128, num_classes=1,
                 cnn_kernel_sizes=[1,3,5]):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            resnet_channels=resnet_channels,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            gru_hidden=gru_hidden,
            cnn_kernel_sizes=cnn_kernel_sizes
        )

        # 计算融合后的特征维度
        # 假设输入序列长度为201（需与实际一致），经过ResNet后长度不变
        # 多尺度CNN分支：每个分支输出特征数 = cnn_channels * (seq_len//2)
        # 假设三个分支，kernel_sizes=[1,3,5]，每个分支输出相同维度
        # 总维度 = 3 * (cnn_channels * (seq_len//2)) + (gru_hidden*2)
        # 这里用占位符，实际可在forward中动态计算，但为静态定义，需知道seq_len
        # 由于我们使用DummyDataset固定seq_len=31，可预先计算
        # 但为了灵活，我们在forward中动态获取seq_len
        # 分类头输入维度不确定，故先设一个大的全连接层，实际运行时特征维度会确定
        self.classifier = nn.Sequential(
            nn.Linear(3 * cnn_channels * (201//2) + gru_hidden*2, 128),  # 占位，运行时动态替换
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        # 动态获取特征维度，如果与预设不符则动态调整（仅第一次运行）
        if not hasattr(self, 'fc_in_features') or self.fc_in_features != features.size(1):
            self.fc_in_features = features.size(1)
            # 替换第一层全连接
            self.classifier[0] = nn.Linear(self.fc_in_features, 128).to(features.device)
        out = self.classifier(features)
        return out

def compile_model(model, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, criterion


if __name__ == "__main__":
    # 设置模型参数
    input_channels = 4  # 对应A, C, G, U
    resnet_channels = 32
    cnn_channels = 64
    lstm_hidden_size = 128
    gru_hidden_size = 128
    num_classes = 1  # 二分类任务
    batch_size = 8
    seq_length = 31  # 根据之前的处理，设定RNA序列的长度为31

    # 创建DeepRNAac4C模型
    model = DeepRNAac4C(input_channels, resnet_channels, cnn_channels, lstm_hidden_size, gru_hidden_size, num_classes)

    # 随机生成一个输入Tensor，模拟一个批次的RNA序列
    x = torch.randn(batch_size, input_channels, seq_length)

    # 模型前向传播
    output = model(x)

    # 打印输出的形状，应该是 (batch_size, num_classes)
    print(f"Output shape: {output.shape}")

    # 编译模型
    optimizer, criterion = compile_model(model)
    print(f"Model compiled with optimizer {optimizer} and loss function {criterion}.")


