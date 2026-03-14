import torch
import torch.nn.functional as F


# 流的核心实现，主要包括采样，流的定义，速度向量的定义，loss的定义
class RectifiedFlow:

    # 图像生成一个迭代公式 ODE f(t+dt) = f(t) + dt*f'(t)
    def euler(self, x_t, v, dt):
        """ 使用欧拉方法计算下一个时间步长的值
            
        Args:
            x_t: 当前的值，维度为 [B, C, H, W]
            v: 当前的速度，维度为 [B, C, H, W]
            dt: 时间步长
        """
        x_t = x_t + v * dt

        return x_t

    def create_flow(self, x_1, t, x_0=None):
        """ 
        创建一个flow流, 表示变换方式
        使用x_t = t * x_1 + (1 - t) * x_0公式构建x_0到x_1的流

            X_1是原始图像 X_0是噪声图像（服从标准高斯分布）
            
        Args:
            x_1: 原始图像，维度为 [B, C, H, W]
            t: 一个标量，表示时间，时间范围为 [0, 1]，维度为 [B]
            x_0: 噪声图像，维度为 [B, C, H, W]，默认值为None
            
        Returns:
            x_t: 在时间t的图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
        
        """
        if x_0 is None:
            x_0 = torch.randn_like(x_1) # 来自噪声

        t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_t = t * x_1 + (1 - t) * x_0
        # 返回初始采样的噪声图 和 在时间t的图像
        return x_t, x_0
    
    def mse_loss(self, v, x_1, x_0):
        """ 
        计算RectifiedFlow的损失函数
        L = MSE(x_1 - x_0 - v(t))  匀速直线运动

        Args:
            v: 速度，维度为 [B, C, H, W]
            x_1: 原始图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
        """
        # 个人理解：速度向量场和时间长度无关，本来就是一个flow的t求导
        loss = F.mse_loss(x_1 - x_0, v) # [B]
        return loss


if __name__ == '__main__':
    rectified_flow = RectifiedFlow()
    x_1 = torch.randn(1, 3, 64, 64)
    t = torch.tensor([0.5])
    x_t, x_0 = rectified_flow.create_flow(x_1, t)
    print(x_t.shape, x_0.shape)