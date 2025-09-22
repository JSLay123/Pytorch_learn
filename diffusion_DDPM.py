import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve

s_curve, _ = make_s_curve(10**4, noise=0.1)
s_curve = s_curve[:, [0,2]]/10.0
print(np.shape(s_curve))        # (10000, 2)\

# data = s_curve.T        # (2, 10000)
#
# fig, ax = plt.subplots()
# ax.scatter(*data, color='red', edgecolors='white')
#
# ax.axis('off')

dataset = torch.tensor(s_curve).float()

num_steps = 100

# 指定每一步的beta
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas)        # 由0-1增加的序列 ,[num_steps]

# 计算alpha alpha_prod alpha_prod_previous alpha_bar_sqrt等 变量
# [100]
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)        # alpha 的连乘
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == alphas_bar_sqrt.shape == one_minus_alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape

# 前向传播过程
# 计算任意时刻的x的采样值,基于x_0的参数重整化技巧
def q_x(x_0, t):
    """基于输入x_0,得到任意时刻t的x_t"""
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return x_0 * alphas_t + alphas_1_m_t * noise

# 噪声预测网络
# x_0, t 为输入, 输出预测噪声
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion,self).__init__()

        self.linear = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
             ]
        )

        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x_0, t):
        x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linear[2*idx](x)
            x += t_embedding
            x = self.linear[2*idx+1](x)
        x = self.linear[-1](x)

        return x

# 损失函数
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t采样计算loss"""
    batch_size = x_0.shape[0]
    # 随机采样时刻t
    t = torch.randint(0, n_steps, size=(batch_size//2,))
    t = torch.cat([t, n_steps-1-t], dim=0)
    t = t.unsqueeze(-1)

    # x0的系数
    a = alphas_bar_sqrt[t]
    # eps的系数
    am1 = one_minus_alphas_bar_sqrt[t]

    # 生成随机噪声eps
    e = torch.randn_like(x_0)

    # 构造模型的输入
    x = x_0 * a + am1 * e

    # 送入模型,计算t时刻的噪声的预测
    output = model(x, t.squeeze(-1))

    # 与真实噪声一起计算误差,求平均值
    return (e - output).square().mean

# 逆扩散过程的采样函数,inference过程
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从x_t采样t时刻的重构值"""
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x,t)

    mean = (1 / (1-betas[t]).sqrt()) * (x - (coeff * eps_theta))
    # 重参数技巧
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """从 x_t 逐步去噪 x_t-1, x_t-2, ... , x_0"""
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


# ---------------------------------训练模型-------------------------------

class EMA():
    """构建一个参数平滑器"""
    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

print("training...")

batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epoch = 4000

# 输入是x_0和step
model = MLPDiffusion(num_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(num_epoch):
    for idx, batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
