import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------cross_entropy loss 交叉熵损失------------------
# 衡量两个分布之间的差异

# logist shape(BS,NC)
batch_size = 2
num_class = 4

logits = torch.randn(batch_size, num_class)
target = torch.randint(num_class, size=(batch_size,))   #delta 目标分布
target_logits = torch.randn(batch_size, num_class)   #非 delta 目标分布

ce_loss_fn = nn.CrossEntropyLoss()
# input 和 target之间少一个类别C维度， ： [N,C]->[N], [N,C,d1,d2]->[N,d1,d2]
ce_loss1 = ce_loss_fn(logits, target)
ce_loss2 = ce_loss_fn(logits, target_logits)


# --------------------NLL loss 负对数似然损失------------------------
