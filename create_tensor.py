import torch
import numpy as np

# from a list
# torch.tensor()
a = [[1, 2], [3., 4]]
x_data = torch.tensor(a)
print(x_data)           #tensor([[1., 2.],[3., 4.]]) 有浮点型则全部为浮点, tensor中只能一种类型
print(type(x_data), x_data.dtype) # <class 'torch.Tensor'> , 数据类型默认torch.float32

# from numpy array
# torch.from_numpy() / torch.tensor()
np_arr = np.random.normal((2,3))
print(np_arr, type(np_arr), np_arr.shape)     # [1.42991156 0.83594287] <class 'numpy.ndarray'> , (2,)
x_data = torch.tensor(np_arr)
print(type(x_data), x_data.dtype, x_data.shape) # <class 'torch.Tensor'> torch.float64, torch.Size([2])

x_np = torch.from_numpy(np_arr) # 同上

# from another tensor
# ones_like 基于已有张量的形状和数据类型，创建一个全0张量
x_t = torch.tensor([[1,2.], [3,4]])
x_ones = torch.ones_like(x_t)           # 含浮点则全为float
x_rand = torch.rand_like(x_t, dtype=torch.float)
print(x_ones, x_rand) #tensor([[1, 1],[1, 1]]) 、 tensor([[0.6135, 0.2873],[0.7122, 0.9935]])

# with shape
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.rand(shape)
zeros_tensor = torch.zeros(shape)

# 属性
# a.dtype, a.shape, a.device
print(zeros_tensor.device)  # device(type='cpu')默认在cpu上创建

# from cpu to gpu
if torch.cuda.is_available():
    zeros_tensor = zeros_tensor.to('cuda')
    print(zeros_tensor.device) # cuda:0

# torch.is_tensor(x)
print(torch.is_tensor(zeros_tensor)) #True

# torch.is_complex() 判断是否是tensor.bool
print(torch.is_complex(zeros_tensor))

# is_nonzero 是否是单一的一个张量且非0
print(torch.is_nonzero(torch.tensor([0.0]))) # False
# torch.is_nonzero(torch.tensor([1.0])) # True
# torch.is_nonzero(torch.tensor([2.0, 3,])) # False

# torch.numel() 查看一个张量中所有元素的数量,shape的连乘
print(torch.numel(zeros_tensor))    # 6

# set data type 设置默认数据类型
# torch.set_default_tensor_type(torch.DoubleTensor)

# range，arange
# arange不包含end，float32，range包含start、end，float64
a = torch.arange(5)
print(a)           # tensor([0, 1, 2, 3, 4])

r = torch.range(1,5)
print(r)

# linspace， logspcae

# torch.eye
print(torch.eye(3))
"""print(torch.eye(3))
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
print(torch.eye(2,3))
tensor([[1., 0., 0.],
        [0., 1., 0.]])"""

# torch.full(size, data), torch.full_like()
print(torch.full((2,3), 5.6))



