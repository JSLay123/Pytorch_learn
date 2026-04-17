import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super(LoRALinear, self).__init__()
        # 1. 模拟原始线性层并冻结
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # 2. 定义 LoRA 权重 (注意形状适配权重矩阵 [out, in])
        # A: [r, in] -> 降维
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        # B: [out, r] -> 升维
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        
        self.reset_parameters()

    def reset_parameters(self):
        # A 使用 Kaiming 初始化，B 初始化为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 计算权重增量 dW = B @ A -> [out, in]
        # 注意：这里是权重之间的矩阵乘法
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        
        # 合并权重 W = W0 + dW
        combined_weight = self.linear.weight + delta_w
        
        # 使用合并后的权重进行线性运算
        # F.linear(x, w) 内部执行的是 x @ w.T
        return F.linear(x, combined_weight, self.linear.bias)

class LoRASelfAttention(nn.Module):
    def __init__(self, embed_dim, r=8):
        super().__init__()
        # 对 Q, K, V 全部应用这种形式的 LoRA
        self.q_proj = LoRALinear(embed_dim, embed_dim, r=r)
        self.k_proj = LoRALinear(embed_dim, embed_dim, r=r)
        self.v_proj = LoRALinear(embed_dim, embed_dim, r=r)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        # 这里的投影内部已经包含了 W0 + BA 的逻辑
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 缩放点积注意力 (Scaled Dot-Product Attention)
        d_k = q.size(-1)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        output = attn_probs @ v
        return self.out_proj(output)

# --- 测试运行 ---
dim = 128
lora_attention = LoRASelfAttention(embed_dim=dim, r=4)
sample_input = torch.randn(1, 10, dim)
output = lora_attention(sample_input)

print(f"输入形状: {sample_input.shape}")
print(f"输出形状: {output.shape}")
# 检查参数：只有 lora_A, lora_B 以及 out_proj 应该有梯度
trainable_params = [n for n, p in lora_attention.named_parameters() if p.requires_grad]
print(f"可训练参数: {trainable_params}")