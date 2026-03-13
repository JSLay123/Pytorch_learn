# 针对手写数字数据集MNIST的Rectified Flow U-Net模型

import torch
import torch.nn as nn

class Downsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16, downsample=False):
        super(Downsample_layer, self).__init__()\
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()
        # 线性层，用于时间编码换通道 [B, dim] -> [B, in_channels]
        self.time_emb = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

        # 下采样
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, t_emb):
        # x : [B, C, H, W]
        # t_emb : [B, time_emb_dim]
        res = x
        x += self.time_emb(t_emb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.conv_shortcut is not None:
            res = self.conv_shortcut(res)

        x = x+res

        if self.downsample:
            x = self.pool(x)

        return x
    
class Upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16, upsample=False):
        super(Upsample_layer, self).__init__()\
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()
        # 线性层，用于时间编码换通道 [B, dim] -> [B, in_channels]
        self.time_emb = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

        # 上采样
        self.upsample = upsample
        if self.upsample:
            self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, t_emb):
        # x : [B, C, H, W]
        # t_emb : [B, time_emb_dim]
        if self.upsample:
            x = self.upsample(x)
        res = x
    
        x += self.time_emb(t_emb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.conv_shortcut is not None:
            res = self.conv_shortcut(res)

        x = x + res
        return x
    
class Middle_layer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(Middle_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

        # 线性层，用于时间编码换通道
        self.time_emb = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        res = x

        x += self.time_emb(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res

        return x
    
class Flow_Unet(nn.Module):
    """两个下采样block和两个上采样block,  一个middle layer"""
    def __init__(self, base_channels = 16, time_emb_dim = None):
        super(Flow_Unet, self).__init__()

        if time_emb_dim is None:
            self.time_emb_dim = base_channels
        self.base_channels = base_channels
        # 黑白图是单通道的，所以输入通道数是1
        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        self.downsample_block_1 = nn.ModuleList([
            Downsample_layer(base_channels, base_channels*2, time_emb_dim=self.time_emb_dim, downsample=False),
            Downsample_layer(base_channels*2, base_channels*2, time_emb_dim=self.time_emb_dim, downsample=False),
        ])

        self.maxpool1 = nn.MaxPool2d(2)

        self.downsample_block_2 = nn.ModuleList([
            Downsample_layer(base_channels*2, base_channels*4, time_emb_dim=self.time_emb_dim, downsample=False),
            Downsample_layer(base_channels*4, base_channels*4, time_emb_dim=self.time_emb_dim, downsample=False),
        ])

        self.maxpool2 = nn.MaxPool2d(2)

        self.middle_block = Middle_layer(base_channels * 4, base_channels * 4, time_emb_dim=self.time_emb_dim)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample_block_1 = nn.ModuleList([
            Upsample_layer(base_channels * 8, base_channels * 2, time_emb_dim=self.time_emb_dim, upsample=False),
            Upsample_layer(base_channels * 2, base_channels * 2, time_emb_dim=self.time_emb_dim, upsample=False),
        ])

        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample_block_2 = nn.ModuleList([
            Upsample_layer(base_channels * 4, base_channels, time_emb_dim=self.time_emb_dim, upsample=False),
            Upsample_layer(base_channels, base_channels, time_emb_dim=self.time_emb_dim, upsample=False),
        ])
        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)

    def time_embedding(self, t, dim):
        """对时间进行正弦函数的编码，单一维度
       目标：让模型感知到输入x_t的时刻t
       实现方式：多种多样
       输入x：[B, C, H, W] x += temb 与空间无关的，也即每个空间位置（H, W）,都需要加上一个相同的时间编码向量[B, C]
       假设B=1 t=0.1
       1. 简单粗暴法
       temb = [0.1] * C -> [0.1, 0.1, 0.1, ……]
       x += temb.reshape(1, C, 1, 1)
       2. 类似绝对位置编码方式
       本代码实现方式
       3. 通过学习的方式（保证T是离散的0， 1， 2， 3，……，T）
       temb_learn = nn.Parameter(T+1, dim)
       x += temb_learn[t, :].reshape(1, C, 1, 1)
       
       
        Args:
            t (float): 时间，维度为[B]
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的时间，维度为[B, dim]  输入是[B, C, H, W]
        """
        # 生成正弦编码
        # 把t映射到[0, 1000]
        t = t * 1000
        # 10000^k k=torch.linspace……
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)
    
    def label_embedding(self, y, dim):
        """对类别标签进行编码，同样采用正弦编码

        Args:
            y (torch.Tensor): 图像标签，维度为[B] label:0-9
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的标签，维度为[B, dim]
        """
        y = y * 1000

        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(y.device)
        sin_emb = torch.sin(y[:, None] / freqs)
        cos_emb = torch.cos(y[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)
    
    def forward(self, x, t, y=None):
        """前向传播

        Args:
            x (torch.Tensor): 输入图像，维度为[B, C, H, W]
            t (torch.Tensor): 时间，维度为[B]
            y (torch.Tensor): 图像标签，维度为[B],目前是数字

        Returns:
            torch.Tensor: 输出图像，维度为[B, C, H, W]
        """
        x = self.conv_in(x)
        temb = self.time_embedding(t, self.base_channels)
        if y is not None:
            if len(y.shape) == 1:
                # label编码，-1表示无条件生成，仅用于训练区分，推理的时候不需要
                # 把y中等于-1的部分找出来不进行任何编码，其余的进行编码
                yemb = self.label_embedding(y, self.base_channels)
                # 把y等于-1的index找出来，然后把对应的y_emb设置为0
                yemb[y == -1] = 0.0
                temb += yemb
            else:  # 文字版本
                pass

        for layer in self.downsample_block_1:
            x = layer(x, temb)
        # skip
        x1 = x
        x = self.maxpool1(x)
        for layer in self.downsample_block_2:
            x = layer(x, temb)
        x2 = x
        x = self.maxpool2(x)

        # 中间层
        x = self.middle_block(x, temb)

        # 上采样
        x = torch.cat([self.upsample1(x), x2], dim=1)
        for layer in self.upsample_block_1:
            x = layer(x, temb)
        x = torch.cat([self.upsample2(x), x1], dim=1)
        for layer in self.upsample_block_2:
            x = layer(x, temb)

        x = self.conv_out(x)
        return x
    
if __name__ == '__main__':
    device = 'cuda'
    model = Flow_Unet()
    model = model.to(device)
    x = torch.randn(2, 1, 28, 28).to(device)
    t = torch.randn(2).to(device)
    y = torch.tensor([1, 2]).to(device)

    out = model(x, t, y)
    print(out.shape)
    # torch.Size([2, 16, 28, 28])