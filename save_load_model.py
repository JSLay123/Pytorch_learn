# saving and loading models for inference
"""
    保存整个训练完以后的模型或模型的参数，可以直接用于后续推理
    只需保存 模型架构和 训练好的权重，无需优化器状态、epoch 等信息
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# model.state_dict()
# 存储一个模型的 parameters and buffers

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc1 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# -------------------------------保存模型的参数------------------------------
# 假设训练完毕，现在需要保存模型训练完以后的参数
PATH = "state_dict_model.pt"
# save
# state_dict(）包含了所有子网络
torch.save(net.state_dict(), PATH)

# 假设要给一个同样的网络导入之前已经训练完以后的参数
# load
#
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()    # 设置为非训练


# -------------------------------保存整个模型------------------------------
# 显然占用空间更大
PATH = "entire_model.pt"
# save
torch.save(net, PATH)

# load
model = torch.load(PATH)
model.eval()


# saving and loading models for inference
# 保存整个训练完以后的模型或模型的参数，可以直接用于后续推理
# 只需保存 模型架构和 训练好的权，无需优化器状态、epoch 等信息

# 而保存为checkpoint则用于 暂停并恢复训练（如训练中途崩溃、调参后继续训练）。
# 需要保存完整训练状态，包括模型、优化器、epoch、loss 等。