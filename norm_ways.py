import torch
import torch.nn as nn

batch_size = 2
time_steps = 3
embedding_dim = 4

# -----------------------------batch norm 批量归一化---------------------------
# 输入维度（N， L， C）
inputx = torch.randn(batch_size, time_steps, embedding_dim)

# pytorch的API实现
batch_norm_op = torch.nn.BatchNorm1d(embedding_dim, affine=False)
# 输入格式要求维度（N， L， C），即特征维度channel 第 1 维
bn_y = batch_norm_op(inputx.transpose(-1,-2)).transpose(-1,-2)

# 手写batch_norm
bn_mean = inputx.mean(dim=(0,1), keepdim=True)
# 输出形状 [1,1,embedding_dim]
bn_std = inputx.std(dim=(0,1), unbiased=False, keepdim=True)
verify_bn_y = (inputx - bn_mean) / (bn_std + 1e-5)

# -----------------------------layer norm 层归一化---------------------------
# 理解为每一个step 的 embedding内部做归一化
# pytorch的API实现
layer_norm_op = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False)
ln_y = layer_norm_op(inputx)

# 手写layer_norm
ln_mean = inputx.mean(dim=-1, keepdim=True)
ln_std = inputx.std(dim = -1, keepdim=True, unbiased=False)# 有偏估计
# torch.Size([2, 3, 1]) torch.Size([2, 3, 1])
verify_ln_y = (inputx - ln_mean) / (ln_std + 1e-5)

# instance norm
# group norm
# weight norm