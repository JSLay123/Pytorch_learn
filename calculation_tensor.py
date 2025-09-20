import torch

# cat 拼接，在维度数不同的那个维度上进行拼接
a = torch.rand((2,3))
b = torch.zeros((2,2))
print(torch.cat([a, b], 1))     # torch.Size([2, 5])

# chunk 默认按照第0个维度分割
c , d = torch.chunk(a, chunks=2, dim=0)
print(c, d)     # tensor([[0.1512, 0.7460, 0.1031]]) tensor([[0.2447, 0.0977, 0.4466]])

c , d = torch.chunk(a, chunks=2, dim=1)
print(c, d)
#tensor([[0.1512, 0.7460],[0.2447, 0.0977]])
# tensor([[0.1031],[0.4466]])

# gather 从原tensor中获取指定dim和指定index的数据
t = torch.tensor([[1,2], [3,4]])
t = torch.gather(t, 1, torch.tensor([[0,0], [1,0]]))
print(t)

# reshape()
print(t.reshape(4))     # tensor([1, 1, 4, 3])
print(t.reshape(-1,))   # tensor([1, 1, 4, 3])

# scatter_， 原位操作
src = torch.arange(1, 11).reshape((2,5))
index = torch.tensor([[0, 1, 2, 0]]) # 索引(0,0)（0,1）(0,2) (0,0)
# 沿着新tensor的第0维，依次放上src上对应索引的元素
torch.zeros(2,5,dtype = src.dtype).scatter_(0, index, src)

# split(),默认沿着0维，分成几个块，也可以传入列表，分成列表中指定的几个块
# chunk()只能均分，多用split()
src = torch.arange(1, 11).reshape((5,2))
print(src.split(2))
# print(src.split(2))
# (tensor([[1, 2],
#         [3, 4]]),
#  tensor([[5, 6],
#         [7, 8]]),
#  tensor([[ 9, 10]]))

# squeeze（）
# 将维度为1的维度去除,也可以指定某个为1的维度
# if input is of shape: (A×1×B×C×1×D) then the input.
# squeeze() will be of shape: (A×B×C×D).

# stack
# 将两个形状相同的张量堆叠起来
a = torch.zeros((3,2))
b = torch.rand((3,2))
c = torch.stack([a,b])
print(c, c.shape)       # torch.Size([2, 3, 2])

d = torch.stack([a,b], dim=1)
print(d, d.shape)       # torch.Size([3, 2, 2])

# torch.take()
e = torch.take(b,torch.tensor([0,2,5])) # 将input当作一位张量进行索引输出
print(e)


# 复制tile
# torch.tile()
# >>> x = torch. tensor([1, 2, 3])
# >>> x. tile((2,))
# tensor([1, 2, 3, 1, 2, 3])
# >>> y = torch. tensor([[1, 2], [3, 4]])
# >>> torch. tile(y, (2, 2)) # 第一个维度复制2分，第二个维度复制2分
# tensor([[1, 2, 1, 2],
#         [3, 4, 3, 4],
#         [1, 2, 1, 2],
#         [3, 4, 3, 4]])
x = torch. tensor([1, 2, 3])
print(x. tile((2,1)))
#tensor([[1, 2, 3],
#        [1, 2, 3]])


# 转置 transpose()
x = torch. tensor([[1, 2, 3], [4, 5, 6]]) # (2,3)
x = x .transpose(0, 1)         # (3,2)
print(x)

# torch.unbind()   将张量沿0,或1 维度 输出成包含多个张量的元组
torch.unbind(x)         # (tensor([1, 2, 3]), tensor([4, 5, 6]))

# unsqueeze()
# 在某一个指定维度上新增加一个维度

# where
# torch.where(x>1,x,y)
# x中满足条件的元素保留，不满足条件的元素的位置用y替代
""">>> x = torch. randn(3, 2)
>>> y = torch. ones(3, 2)
>>> x
tensor([[-0.4620,  0.3139],
        [ 0.3898, -0.7197],
        [ 0.0478, -0.1657]])
>>> torch. where(x > 0, 1.0, 0.0)
tensor([[0., 1.],
        [1., 0.],
        [1., 0.]])
>>> torch. where(x > 0, x, y)
tensor([[ 1.0000,  0.3139],
        [ 0.3898,  1.0000],
        [ 0.0478,  1.0000]])
>>> x = torch. randn(2, 2, dtype=torch. double)
>>> x
tensor([[ 1.0779,  0.0383],
        [-0.8785, -1.1089]], dtype=torch. float64)
>>> torch. where(x > 0, x, 0.)
tensor([[1.0779, 0.0383],
        [0.0000, 0.0000]], dtype=torch. float64)"""


# 设置随机数种子
torch.manual_seed(0)        # 同时将cuda的也设置好了，不过numpy的种子还需要设定

# torch.bernoulli() # 对应位置都是0或1

# 高斯分布
torch.normal(mean = torch.tensor([1,2,3]), std = torch.tensor([0.1,0.1,0.1]))
# 也可以共享均值，方差不一样
# 共享方差，均值不一样
# 都共享，设置size
torch.normal(1, 0.1, (2,3))

# rand()        [0,1)均匀分布采样
# randint()     [min,max)
# rand()        01之间的正态分布，只需要传入size
# randperm()    返回一个随机排列
#>>> torch. randperm(4) # tensor([2, 1, 0, 3])

# torch.view()
data = torch.arange(0,10,dtype=torch.float32).view(-1,2)
print(data.shape)       # torch.Size([5, 2])