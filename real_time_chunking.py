import torch
import time

def get_action(self, o_next):
        """
        [Algorithm 1: GETACTION]
        由控制器以固定频率调用 (如 30Hz)
        """
        with self.mutex:
            # 1. 更新观测
            self.o_cur = o_next
            
            # 2. 通知后台线程观测已更新 (如果是新的一步)
            self.t += 1
            self.cond.notify()
            
            # 3. 返回当前时间步的动作
            # 注意：如果 t 超出了 H，通常需要处理异常，
            # 但 RTC 设计保证新块会及时替换
            if self.t > self.H:
                current_action = self.A_cur[-1] # Fallback
            else:
                current_action = self.A_cur[self.t - 1]
                
        return current_action


def inference_loop(self):
    """
    [Algorithm 1: INFERENCELOOP]
    后台循环线程，负责生成新的动作块
    """
    while self.running:
        with self.cond:
            # 1. 等待直到满足最小执行时间 s_min 
            # 同时也确保有新的观测数据进来
            while self.t < self.s_min:
                self.cond.wait()
            
            # 2. 获取用于推理的快照数据
            s = self.t  # 已执行的步数，也是开始推理时的步数
            # 提取上一块中剩余未执行的部分 A_prev (用于Inpainting)
            # A_prev = A_cur[s : H]
            if s < self.H:
                A_prev = self.A_cur[s:].clone()
            else:
                A_prev = torch.empty(0) # 上一块已耗尽
            
            obs = self.o_cur
            
            # 3. 估计延迟 d 
            d_est = max(self.delay_queue)
        
        # --- 释放锁进行繁重的推理计算 (With M released) ---
        
        start_time = time.time()
        
        # 4. 执行引导式推理 (核心算法)，下文有单独实现
        A_new = self.guided_inference(obs, A_prev, d_est, s)
        
        inference_time = time.time() - start_time
        
        # --- 重新获取锁更新状态 ---
        with self.mutex:
            # 5. 只要新块可用，立即交换 (Swap)
            self.A_cur = A_new
            
            # 6. 重置 t，使其指向新块的正确位置
            # 新块生成花费了实际延迟时间，因此我们需要跳过已经过去的时间步
            # 实际延迟计算逻辑可能需要根据具体的 system clock 调整，这里简化为 s
            # 论文伪代码写的是 t = t - s，意味着新块从索引 t-s 开始用 (通常是0附近，取决于对齐)
            self.t = self.t - s 
            
            # 7. 记录实际延迟 (转换为时间步)
            # 假设 control_dt 是控制周期，例如 0.02s
            control_dt = 0.02 
            actual_delay_steps = int(inference_time / control_dt)
            self.delay_queue.append(actual_delay_steps)


def guided_inference(self, obs, A_prev, d, s):
        """
        [Algorithm 1: GUIDEDINFERENCE]
        基于流匹配 (Flow Matching) 和 Inpainting 的生成过程
        """
        H = self.H
        action_dim = self.A_cur.shape[1]
        
        # 1. 计算 Soft Masking 权重 W [Eq. 5] [cite: 162]
        # W 决定了我们多大程度上通过 A_prev 来约束新动作
        W = torch.zeros(H, action_dim)
        
        # 重叠区域长度
        overlap_len = A_prev.shape[0] 
        
        # 填充 A_prev 到长度 H (右侧补零，虽然补零部分权重为0不影响)
        A_prev_padded = torch.zeros(H, action_dim)
        
        if overlap_len > 0:
            A_prev_padded[:overlap_len] = A_prev
 
        for i in range(H):
            if i < d:
                # 冻结区域：必须完全匹配上一块，因为这些时刻在推理完成前就会被执行
                W[i] = 1.0
            elif d <= i < H - s:
                # 中间过渡区域：指数衰减权重，允许平滑修正
                # c_1 是衰减系数
                c_1 = (H - s - i) / (H - s - d + 1 + 1e-6) 
                W[i] = c_1
            else:
                # 自由生成区域：新产生的未来动作，权重为 0
                W[i] = 0.0
               # 2. 初始化噪声 A^0 ~ N(0, I)
               
        A_tau = torch.randn(H, action_dim)
        
        # 3. 迭代去噪循环 (Euler Integration) 
        dt = 1.0 / self.n
        
        for k in range(self.n):
            tau = k / self.n # 当前时间步 [0, 1]
            t_tensor = torch.tensor(tau).float()
            
            # 开启梯度计算，因为我们需要对 input 求导来计算 Guidance
            A_tau.requires_grad_(True)
            
            # 预测速度场 v
            v_pred = self.model(A_tau, obs, t_tensor)
            
            # --- 计算 Guidance (Inpainting) ---
            
            # 估计最终生成的动作 A_hat^1 [Eq. 3]
            # A_hat^1 = A_tau + (1 - tau) * v_pred
            A_hat_1 = A_tau + (1 - tau) * v_pred
            
            # 计算加权误差项 e
            # e = (A_prev - A_hat_1) * W
            # 注意：这里我们只关心重叠部分的误差，右侧补0了所以不影响
            error = (A_prev_padded - A_hat_1) * W
            
            # 计算 Guidance 梯度 g [Eq. 2 & Algorithm 1 line 27]
            # g = e * d(A_hat_1)/d(A_tau)
            # 这本质上是 error 对 A_tau 的梯度
            # 我们可以通过反向传播计算这个向量-雅可比积
            
            # 这里的 Loss 本质上是 || (A_prev - A_hat_1) * sqrt(W) ||^2 的导数相关项
            # 简化的实现方式是直接对 error 求和后 backward
            # 但论文公式 explicit 写的是 Vector-Jacobian Product
            
            # 技巧：由于我们需要 grad(A_hat_1, A_tau) * error
            # 这等价于 torch.autograd.grad(A_hat_1, A_tau, grad_outputs=error)
            g = torch.autograd.grad(outputs=A_hat_1, inputs=A_tau, 
                                    grad_outputs=error, retain_graph=False)[0]
            # 计算 Guidance 权重限制 
            # term = min(beta, (1-tau)/(tau * r_tau^2))
            # r_tau^2 定义见 Eq. 4: (1-tau)^2 / (tau^2 + (1-tau)^2)
            if tau == 0:
                weight_clip = self.beta # 避免除以0，第一步直接用 beta
            else:
                r_tau_sq = ((1-tau)**2) / (tau**2 + (1-tau)**2)
                val = (1 - tau) / (tau * r_tau_sq)
                weight_clip = min(self.beta, val)   
                
            # 4. 更新步 (Integration Step) [Eq. 1 & Algorithm 1 line 28]
            # A^{tau+1/n} = A^tau + 1/n * (v + weight * g)
 
            # dt 对应 1/n
            dt = 1.0 / self.n            
 
            with torch.no_grad():
                updated_v = v_pred + weight_clip * g
                A_tau = A_tau + dt * updated_v
                A_tau = A_tau.detach() #以此作为下一步的输入，切断图
        
        return A_tau