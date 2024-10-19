import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
 
# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # 第一个全连接层
        self.fc2 = nn.Linear(64, 32)  # 第二个全连接层
        self.fc3 = nn.Linear(32, num_classes)  # 输出层，维度为类别数量
 
    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU激活函数
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层不需要激活函数，因为我们会在损失函数中应用Softmax
        return x
 
# 模型实例化
input_size = 128  # 假设输入数据的特征维度为128
num_classes = 10  # 假设有10个类别
model = SimpleNN(input_size, num_classes)
 
# 打印模型结构
print(model)
 
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss包含了Softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 假设有一些数据用于训练
# inputs: shape (batch_size, input_size)
# labels: shape (batch_size)
inputs = torch.randn(32, input_size)  # 32个样本，输入特征维度为128
labels = torch.randint(0, num_classes, (32,))  # 32个样本的标签，范围在0到9之间
 
# 训练步骤
outputs = model(inputs)  # 前向传播
loss = criterion(outputs, labels)  # 计算损失
optimizer.zero_grad()  # 清空梯度
loss.backward()  # 反向传播
optimizer.step()  # 更新参数
 
# 打印损失
print(f'Loss: {loss.item()}')