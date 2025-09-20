# saving and loading a general checkpoint
"""
    保存为checkpoint则用于 暂停并恢复训练（如训练中途崩溃、调参后继续训练）。
    需要保存完整训练状态，包括模型、优化器、epoch、loss 等。
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

# -----------------save a general checkpoint--------------------------
# 保存为一个字典
EPOCH = 5
PATH  = "model.pth"
LOSS = 0.4

torch.save({
    'epoch' : EPOCH,
    'model_state_dict' : net.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
    'loss' : LOSS,},
    PATH,
)

# -----------------load thre general checkpoint------------------------
# 先下载检查点，然后load对应的参数
model = Net()
optimizer1 = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# or
model.train()

# ----------------保存多个模型时，同理保存为字典，然后下载时根据key去load_state_dict到对应的模型----