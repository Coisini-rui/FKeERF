import torch
import torch.nn as nn
import torch
import torch.nn as nn

# ResNet模块（多层卷积+残差）
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super().__init__()
        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1)
            )
            current_channels = out_channels
        self.relu = nn.ReLU()
        self.residual_conv = nn.Conv1d(self.in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = self.relu(layer(x))
        residual = self.residual_conv(residual)
        return x + residual


# 多尺度CNN分支（并行卷积+池化，最后扁平化）
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1,3,5], pool_size=2):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2),  # 保持长度
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),  # 长度减半
                nn.Dropout(0.3)
            )
            self.branches.append(branch)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)                     # (batch, out_channels, seq_len//2)
            out = out.flatten(start_dim=1)       # (batch, out_channels * (seq_len//2))
            branch_outputs.append(out)
        # 沿特征维度拼接
        return torch.cat(branch_outputs, dim=1)  # (batch, total_features)


# BiLSTM + BiGRU 分支（串联，取最后时间步）
class BiLSTM_BiGRU(nn.Module):
    def __init__(self, input_size, lstm_hidden, gru_hidden, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden*2, gru_hidden, num_layers=num_layers,
                          bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x, _ = self.lstm(x)                     # (batch, seq_len, lstm_hidden*2)
        x, _ = self.gru(x)                       # (batch, seq_len, gru_hidden*2)
        x = x[:, -1, :]                           # 取最后时间步: (batch, gru_hidden*2)
        return x


# 完整特征提取器（符合论文架构）
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=4, resnet_channels=32, cnn_channels=64,
                 lstm_hidden=128, gru_hidden=128, resnet_layers=3,
                 cnn_kernel_sizes=[1,3,5]):
        super().__init__()
        # ResNet（保持序列长度）
        self.resnet = ResNetBlock(input_channels, resnet_channels, num_layers=resnet_layers)

        # 分支1：多尺度CNN
        self.cnn_branch = MultiScaleCNN(resnet_channels, cnn_channels, kernel_sizes=cnn_kernel_sizes)

        # 分支2：BiLSTM+BiGRU
        self.lstm_gru_branch = BiLSTM_BiGRU(resnet_channels, lstm_hidden, gru_hidden)

        # 注意：两个分支的输入都是ResNet的输出，但格式不同：
        # CNN分支需要 (batch, channels, seq_len)
        # LSTM/GRU分支需要 (batch, seq_len, channels)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        # ResNet
        res_out = self.resnet(x)                 # (batch, resnet_channels, seq_len)

        # 分支1：多尺度CNN（直接使用res_out）
        cnn_features = self.cnn_branch(res_out)  # (batch, cnn_total_features)

        # 分支2：BiLSTM+BiGRU（需要转置为(batch, seq_len, channels)）
        lstm_input = res_out.permute(0, 2, 1)    # (batch, seq_len, resnet_channels)
        lstm_features = self.lstm_gru_branch(lstm_input)  # (batch, gru_hidden*2)

        # 特征融合：拼接
        combined = torch.cat([cnn_features, lstm_features], dim=1)
        return combined


if __name__ == "__main__":
    # 设置模型参数
    input_channels = 4  # 对应A, C, G, U
    resnet_channels = 32
    cnn_channels = 64
    lstm_hidden_size = 128
    gru_hidden_size = 128
    batch_size = 8
    seq_length = 31  # 根据之前的处理，设定RNA序列的长度为31

    # 创建FeatureExtractor模型
    model = FeatureExtractor(input_channels, resnet_channels, cnn_channels, lstm_hidden_size, gru_hidden_size)

    # 随机生成一个输入Tensor，模拟一个批次的RNA序列（假设batch_size=8, seq_length=31, input_channels=4）
    x = torch.randn(batch_size, input_channels, seq_length)

    # 前向传播，测试模型能否正常运行
    output = model(x)

    # 打印输出形状，应该是 (batch_size, gru_hidden_size*2) 因为BiGRU是双向的
    print(f"Output shape: {output.shape}")
