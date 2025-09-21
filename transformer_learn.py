import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

# the main and hard part of transformer learning

# 关于word embeding ， 以序列建模为例
# 考虑source sentence 和 target sentence
# 构建序列，序列字符以词表中索引的形式表示

batch_size = 2
# 单词表的大小
max_num_src_words = 8
max_num_tgt_words = 8

# embedding 后的维度， 原文512
model_dim = 8

# 最大序列长度
max_src_seq_len = 5
max_tgt_seq_len = 5
max_position_len = 5

# src_len = torch.randint(2, 5, (batch_size, ))
# tgt_len = torch.randint(2, 5, (batch_size, ))
# print(src_len, tgt_len)

src_len = torch.Tensor([2,4]).to(torch.int32) # 两个batch，第一个长度2, 第二个长度4
tgt_len = torch.Tensor([4,3]).to(torch.int32)

# 原序列 + padding
src_seq = [ F.pad(torch.randint(1,  max_num_src_words, (L,)) , (0, max(src_len) - L))for L in src_len ] # 单词索引构成的句子
# print(src_seq)  # [tensor([6, 6, 0, 0]), tensor([2, 4, 4, 3])]
# 目标序列 + padding
tgt_seq = [ F.pad(torch.randint(1,  max_num_src_words, (L,)) , (0, max(src_len) - L))for L in tgt_len ]
# print(tgt_seq)  # [tensor([1, 4, 3, 4]), tensor([4, 3, 7, 0])]

# 将src 、tgt 处理为tensor格式张量
src_seq = torch.cat([torch.unsqueeze(i ,0) for i in src_seq])
# print(src_seq)
tgt_seq = torch.cat([torch.unsqueeze(i ,0) for i in tgt_seq])
# print(tgt_seq)
"""
原句子和目标句子，(2,4)
tensor([[2, 6, 0, 0],
        [2, 3, 7, 6]])
tensor([[3, 4, 2, 2],
        [1, 4, 6, 0]])"""

# -----------------------构造word embedding, 这里是一个类 shape(9,8)--------------------------
# Embedding和Linear几乎是一样的，区别就在于：输入不同，一个是输入数字，一个是输入one-hot向量。
src_embedding_table = nn.Embedding(max_num_src_words+1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1, model_dim)

src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)

# print(src_embedding.shape, tgt_embedding.shape) # torch.Size([2, 4, 8]) torch.Size([2, 4, 8])

# ------------------------------构造position embedding类--------------------------------------------
class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len, model_dim ):
        super().__init__()
        pos_enc = torch.zeros(max_seq_len, model_dim)      # (5,8)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # (5, 1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim) ) # (4)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)  # (1, max_seq_len, model_dim)

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :]  # 从[1, max_seq_len, d_model]中取[1, x_seq_len, d_model],形状匹配
        return x

# ------------------------------
# 实例化 position embedding
pe_embedding = PositionEncoding(max(src_len), model_dim)
src_pe_embedding = pe_embedding(src_embedding)
trt_pe_embedding = pe_embedding(tgt_embedding)

# -------------------------------构造encoder的self-attention mask--------------------------------
# 让padding的地方的分数约为0
# mask的沙配额 ： [batch_size, max_src_len, max_src_len]，值为 1 或 -inf
# 获取有效位置 # [tensor([1., 1.]), tensor([1., 1., 1., 1.])]
# (2,1,4)
valid_encoder_pos = torch.unsqueeze(
    torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L)), dim=0) for L in src_len]),
    dim=1)
print(valid_encoder_pos)
# tensor([[[1., 1., 0., 0.]],
#         [[1., 1., 1., 1.]]])

# (2,4,4) 表示2个batch，每个batch里面每个单词更其他单词的有效性，将padding的词和位置为无效
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos.transpose(1,2), valid_encoder_pos)
print(valid_encoder_pos_matrix.shape)
# tensor([[[1., 1., 0., 0.],
#          [1., 1., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],
#         [[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]]])
# 变成一个bool型的mask，True代表需要mask，False代表不进行mask
mask_encoder_self_attention = (1 - valid_encoder_pos_matrix).to(bool)

# -----------------------------mask_encoder_self_attention的使用--------------------------------
# 假设有一个socre，形状[2,4,4]
score = torch.randn(batch_size, max(src_len), max(src_len))
# 对padding的地方计算出来的分数给一个-无穷,这样softmax时约为0
masked_score = score.masked_fill(mask_encoder_self_attention, -np.inf)
prob = F.softmax(masked_score, -1)



# -----------------------------构造 cross_attention 的mask---------------------------------------
# Q @ K^T的shape:[batch_size, tgt_seq_len, src_seq_len]
valid_encoder_pos = torch.unsqueeze(
    torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L)), dim=0) for L in src_len]),
    dim=1)

valid_decoder_pos = torch.unsqueeze(
    torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(tgt_len) - L)), dim=0) for L in tgt_len]),
    dim=1)

print(valid_encoder_pos, valid_decoder_pos)     # [2,1,4]
# tensor([[[1., 1., 0., 0.]],
#         [[1., 1., 1., 1.]]])
# tensor([[[1., 1., 1., 1.]],
#         [[1., 1., 1., 0.]]])
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos.transpose(1,2), valid_encoder_pos)
print(valid_cross_pos_matrix.shape)       # # [2,4,4]
# 目标序列各个词对原序列各个词的有效性，一个batch中的三行2列表示 tgt序列第三个word对src序列第二个word的有效性
# tensor([[[1., 1., 0., 0.],
#          [1., 1., 0., 0.],
#          [1., 1., 0., 0.],
#          [1., 1., 0., 0.]],
#         [[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [0., 0., 0., 0.]]])

# 变成一个bool型的mask，True代表需要mask，False代表不进行mask
mask_cross_attention = (1 - valid_cross_pos_matrix).to(bool)
# 同上，对计算出来的分数进行：
# masked_score = score.masked_fill(mask_encoder_self_attention, -np.inf)


# -------------------------------构造decoder的self-attention casual mask--------------------------------
# pad参数的顺序是  从输入张量的最后一个维度开始向前指定，每个维度需要 2 个值（左填充和右填充）
# [2,4,4]
valid_decoder_tri_matrix = torch.cat(
    [torch.unsqueeze(F.pad(torch.tril(torch.ones(L,L)), (0, max(tgt_len) -L, 0, max(tgt_len) -L)), dim=0)
     for L in tgt_len])
# tensor([[[1., 0., 0., 0.],
#          [1., 1., 0., 0.],
#          [1., 1., 1., 0.],
#          [1., 1., 1., 1.]],
#         [[1., 0., 0., 0.],
#          [1., 1., 0., 0.],
#          [1., 1., 1., 0.],
#          [0., 0., 0., 0.]]])

mask_casual_attention = (1 - valid_decoder_tri_matrix).to(bool)
cross_score = torch.randn(batch_size, max(tgt_len), max(tgt_len))
# 对padding的地方计算出来的分数给一个-无穷,这样softmax时约为0
casual_masked_score = cross_score.masked_fill(mask_casual_attention, -np.inf)
cross_prob = F.softmax(casual_masked_score, -1)

# ------------------------------构造scaled self-attention-----------------------------
def scaled_dot_product_attention(Q, K, V, attn_mask):
    score = torch.bmm(Q, K.transpose(-2,-1)) / torch.sqrt(model_dim)
    masked_score = score.masked_fill(attn_mask, -np.inf)
    prob = F.softmax(masked_score, -1)
    context = torch.bmm(prob, V)
    return context