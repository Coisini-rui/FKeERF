import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from model_building import DeepRNAac4C  # 引用模型结构

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型、优化器、损失函数
# 模型参数（与论文一致，但seq_len=31可调整）
input_channels = 4
resnet_channels = 32
cnn_channels = 64
lstm_hidden = 128
gru_hidden = 128
batch_size = 32
num_epochs = 10
seq_len = 31  # 当前DummyDataset使用的长度

# 创建模型
# model = DeepRNAac4C(input_channels, resnet_channels, cnn_channels, lstm_hidden_size, gru_hidden_size, num_classes=1)
model = DeepRNAac4C(
    input_channels=input_channels,
    resnet_channels=resnet_channels,
    cnn_channels=cnn_channels,
    lstm_hidden=lstm_hidden,
    gru_hidden=gru_hidden,
    num_classes=1,
    cnn_kernel_sizes=[1,3,5]   # 三个分支
)

model = model.to(device)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # 适用于二分类任务


# 示例数据集（可以替换成你实际的Dataset）——因为分类方法不一样，所以这里直接用示例数据集做演示证明特征提取可运行
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, seq_len=31, num_classes=1):
        self.size = size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.data = torch.randn(size, 4, seq_len)  # 随机生成数据 (batch, channels, length)
        self.labels = torch.randint(0, num_classes, (size,))  # 随机标签

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 数据加载
train_dataset = DummyDataset(size=2000, seq_len=31, num_classes=1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = DummyDataset(size=500, seq_len=31, num_classes=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 训练函数
def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(data)

        # 计算损失
        loss = criterion(outputs.squeeze(), labels.float())

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 保存预测和真实标签
        all_preds.extend(outputs.squeeze().cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算训练精度
    # preds = (torch.sigmoid(torch.tensor(all_preds)) > 0.5).astype(int)
    preds = (torch.sigmoid(torch.tensor(all_preds)) > 0.5).int().numpy()
    accuracy = accuracy_score(all_labels, preds)

    return running_loss / len(train_loader), accuracy


# 评估函数
def evaluate(model, val_loader, criterion):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            # 前向传播
            outputs = model(data)

            # 计算损失
            loss = criterion(outputs.squeeze(), labels.float())

            running_loss += loss.item()

            # 保存预测和真实标签
            all_preds.extend(outputs.squeeze().cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估精度
    # preds = (torch.sigmoid(torch.tensor(all_preds)) > 0.5).astype(int)
    preds = (torch.sigmoid(torch.tensor(all_preds)) > 0.5).int().numpy()
    accuracy = accuracy_score(all_labels, preds)
    # report = classification_report(all_labels, preds, target_names=["Class 0", "Class 1"])
    report = classification_report(
        all_labels,
        preds,
        labels=[0, 1],  # 明确指定需要报告的类别
        target_names=["Class 0", "Class 1"],
        zero_division=0  # 避免因零样本而警告
    )

    return running_loss / len(val_loader), accuracy, report


# 训练过程
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # 训练
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # 验证
    val_loss, val_acc, val_report = evaluate(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    print(val_report)

    # 每个epoch保存模型
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")


