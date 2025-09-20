import torch

# forward
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b      # torch.Size([3])
# print(z.shape)
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 每个节点处都有一个grad_func
print(z.grad_fn, '\n', loss.grad_fn)
"""<AddBackward0 object at 0x73a97c8f9270> 
 <BinaryCrossEntropyWithLogitsBackward0 object at 0x73a97c8fa0b0>"""

# computing grad
loss.backward()
# 有一个动态计算图，算完一次就清除 （or retain_graph = True）

# loss关于w的梯度，loss关于b的梯度
print(w.grad)
print(b.grad)
"""tensor([[0.3265, 0.2493, 0.3108],
        [0.3265, 0.2493, 0.3108],
        [0.3265, 0.2493, 0.3108],
        [0.3265, 0.2493, 0.3108],
        [0.3265, 0.2493, 0.3108]])
tensor([0.3265, 0.2493, 0.3108])"""

# 取消梯度更新
z = torch.matmul(x, w) + b
print(z.requires_grad)      # True

with torch.no_grad():
    z = torch.matmul(x, w) + b
    print(z.requires_grad)  # False