import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()             # start_dim = 1, end_dim = -1
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

def train_loop(dataloader, model , loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to('cuda'), y.to('cuda')
        # 计算损失
        pred = model(X)
        loss = loss_fn(pred, y)
        # 反向传播
        optimizer.zero_grad()   # # 默认优化器的导数保留
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"当前batch：{current}, 当前Loss：{loss}")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to('cuda'), y.to('cuda')
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"当前测试的loss：{test_loss}，当前的正确率:{correct}\n" )

def main():
    batch_size = 64

    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = NeuralNetwork().cuda()

    learning_rate = 1e-3

    epochs = 20

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

# if __name__ == '__main__':
#     main()


# 补充-------------------------------------------
# nn.Embedding 学习
"""
在PyTorch中，nn.Embedding用来实现词与词向量的映射。
nn.Embedding可以理解为一个没有bias的全连接，具有权重（.weight），形状是(num_words, embedding_dim)
    例如一共有100个词，每个词用16维向量表征，对应的权重就是一个100×16的矩阵。
Embedding的输入形状N×W，N是batch size，W是序列的长度，输出的形状是N×W×embedding_dim。
Embedding的权重是可以训练的
"""