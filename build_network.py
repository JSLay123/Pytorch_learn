import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()             # start_dim = 1, end_dim = -1
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

model = NeuralNetwork().to('cuda')
print(model)

input_img = torch.rand(3, 28, 28).to('cuda')
print(input_img.size())

# Flatten()是保留第一个维度，也就是batch_size维度，其他都展平

# Linear()
#        in_features: int,
#        out_features: int,
#        bias: bool = True

print(model(input_img).shape)       # torch.Size([3, 10])

# 查看网络结构和参数，常用
for name, param in model.named_parameters():
    print(f"Layer : {name} | Size : {param.shape} | Value : {param[:2]} \n")

"""Layer : linear_relu_stack.0.weight | Size : torch.Size([512, 784]) | Value : tensor([[-0.0303, -0.0157, -0.0226,  ..., -0.0239, -0.0293,  0.0083],
        [ 0.0155, -0.0339,  0.0287,  ..., -0.0002,  0.0181, -0.0343]],
       device='cuda:0', grad_fn=<SliceBackward0>) 
Layer : linear_relu_stack.0.bias | Size : torch.Size([512]) | Value : tensor([-0.0324, -0.0304], device='cuda:0', grad_fn=<SliceBackward0>) 
Layer : linear_relu_stack.2.weight | Size : torch.Size([512, 512]) | Value : tensor([[-0.0036, -0.0180, -0.0161,  ..., -0.0196, -0.0146,  0.0158],
        [-0.0084, -0.0338,  0.0437,  ...,  0.0123,  0.0178,  0.0361]],
       device='cuda:0', grad_fn=<SliceBackward0>) 
Layer : linear_relu_stack.2.bias | Size : torch.Size([512]) | Value : tensor([0.0253, 0.0269], device='cuda:0', grad_fn=<SliceBackward0>) 
Layer : linear_relu_stack.4.weight | Size : torch.Size([10, 512]) | Value : tensor([[ 0.0409, -0.0419, -0.0022,  ..., -0.0203, -0.0406, -0.0218],
        [ 0.0257,  0.0415, -0.0380,  ..., -0.0093, -0.0363,  0.0408]],
       device='cuda:0', grad_fn=<SliceBackward0>) 
Layer : linear_relu_stack.4.bias | Size : torch.Size([10]) | Value : tensor([-0.0056,  0.0349], device='cuda:0', grad_fn=<SliceBackward0>) 
"""

# ._modules 返回一个字典，所有的子模块
print(model._modules)
"""{'flatten': Flatten(start_dim=1, end_dim=-1), 'linear_relu_stack': Sequential(
  (0): Linear(in_features=784, out_features=512, bias=True)
  (1): ReLU()
  (2): Linear(in_features=512, out_features=512, bias=True)
  (3): ReLU()
  (4): Linear(in_features=512, out_features=10, bias=True)
)}"""

# _parameters
# _buffer
# 这都是获取当前模块的参数，不能获取其子模块，除非 model.linear_relu_stack[0]._parameters这种指定具体模块

# parameters(), 所有参数,不包含名字，一般多使用.named_parameters()
for i in model.parameters(): # 一个generator
    print(i)

# named_children(), 所有子模块，元组
# 返回包含子模块的迭代器，同时产生模块的名称以及模块本身。
for i in model.named_children(): # 一个generator
    print(i)
"""('flatten', Flatten(start_dim=1, end_dim=-1))
('linear_relu_stack', Sequential(
  (0): Linear(in_features=784, out_features=512, bias=True)
  (1): ReLU()
  (2): Linear(in_features=512, out_features=512, bias=True)
  (3): ReLU()
  (4): Linear(in_features=512, out_features=10, bias=True)
))"""

# model.modules()       展示所有模块，先自身总体，后子模块，全部依次展示
# model.named_modules() 返回网络中所有模块的迭代器，同时产生模块的名称以及模块本身。


# .to() 改变device ， 数据类型
model.to(dtype=torch.double)

# .state_dict() 查看所有的参数,字典
print(model.state_dict())
print(model.state_dict()['linear_relu_stack.4.bias']) # 可以索引具体参数
#tensor([-0.0056,  0.0349,  0.0342,  0.0088,  0.0399, -0.0411,  0.0193,  0.0076,-0.0251, -0.0062], device='cuda:0')

# save and load the model
# 保存模型的权重
import torchvision.models as models
model_vgg = models.vgg16(pretrained = True)
torch.save(model.state_dict(), 'model_weights.pth')

# 推理时，先导入空模型，然后加载参数
model_vgg_empty = models.vgg16()
model_vgg_empty.load_state_dict(torch.load('model_weights.pth'))
model_vgg_empty.eval()      # 推理模式，影响dropout等...

# train() 模型中子模块（所有模块）设置为self.train(True)
# 影响dropout，batch_norm
# model.train()

# 梯度计算
# 模型所有参数.parameters() 的 requires_grad设置为true
# model.requires_grad_()

# 优化器的梯度清零
# optimizer.zero_grad()
