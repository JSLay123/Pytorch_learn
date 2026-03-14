import torch
import os
import yaml
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from rectified_flow_unet import Flow_Unet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from rectified_flow import RectifiedFlow
from reflow_dataset import ReflowDataset


# 1. reflow的训练要从上一个1-rectified flow(v1.1)模型的权重作为预训练权重
def train(config:str):
    """训练reflow模型

    Args:
        config (str): yaml配置文件路径，包含以下参数：
            base_channels (int, optional): MiniUnet的基础通道数，默认值为16。
            epochs (int, optional): 训练轮数，默认值为10。
            batch_size (int, optional): 批大小，默认值为128。
            lr_adjust_epoch (int, optional): 学习率调整轮数，默认值为50。
            batch_print_interval (int, optional): batch打印信息间隔，默认值为100。
            checkpoint_save_interval (int, optional): checkpopint保存间隔(单位为epoch)，默认值为1。
            save_path (str, optional): 模型保存路径，默认值为'./checkpoints'。
            use_cfg (bool, optional): 是否使用Classifier-free Guidance训练条件生成模型，默认值为False。
            device (str, optional): 训练设备，默认值为'cuda'。

    """
    config = yaml.safe_load(open(config, 'rb'))
    base_channels = config.get('base_channels', 16)
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 128)
    lr_adjust_epoch = config.get('lr_adjust_epoch', 50)
    batch_print_interval = config.get('batch_print_interval', 100)
    checkpoint_save_interval = config.get('checkpoint_save_interval', 1)
    save_path = config.get('save_path', './checkpoints')
    use_cfg = config.get('use_cfg', False)
    device = config.get('device', 'cuda')
    # v1.2 reflow增加参数
    lr = config.get('lr', 1e-5)
    img_root_path = config.get('img_root_path', None)
    noise_root_path = config.get('noise_root_path', None)
    checkpoint_path = config.get('checkpoint_path', None)

    # 打印训练参数
    print('Training config:')
    print(f'base_channels: {base_channels}')
    print(f'epochs: {epochs}')
    print(f'batch_size: {batch_size}')
    print(f'learning rate: {lr}')
    print(f'lr_adjust_epoch: {lr_adjust_epoch}')
    print(f'batch_print_interval: {batch_print_interval}')
    print(f'checkpoint_save_interval: {checkpoint_save_interval}')
    print(f'save_path: {save_path}')
    print(f'use_cfg: {use_cfg}')
    print(f'img_root_path: {img_root_path}')
    print(f'noise_root_path: {noise_root_path}')
    print(f'checkpoint_path: {checkpoint_path}')
    print(f'device: {device}')

    # 数据集加载
    transform = Compose([ToTensor()])   # 变成tensor的同时，增加一个维度

    dataset = ReflowDataset(img_root_path, noise_root_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Flow_Unet(base_channels=base_channels).to(device)

    # 优化器加载 Rectified Flow的论文里面有的用的就是AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    # 学习率调整
    scheduler = StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

    # 加载flow
    rf = RectifiedFlow()

    loss_list = []

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        for batch, data in enumerate(dataloader):
            x_1 = data['img']
            print(x_1.size())
            x_0 = data['noise']
            y = data['label']
            t = torch.randn(x_1.size(0))

            x_t, _ = rf.create_flow(x_1, t, x_0)

            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            t = t.to(device) 

            optimizer.zero_grad()

            # 这里做一个数据的复制和拼接，复制原始x_1，把一半的y替换成-1表示无条件生成，这里也可以直接有条件、无条件累计两次计算两次loss的梯度
            # 把有条件生成换为无条件的 50%的概率 [x_t, x_t] [t, t]
            if use_cfg:
                x_t = torch.cat([x_t, x_t.clone()], dim=0)
                t = torch.cat([t, t.clone()], dim=0)
                y = torch.cat([y, -torch.ones_like(y)], dim=0)
                x_1 = torch.cat([x_1, x_1.clone()], dim=0)
                x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                y = y.to(device)
            else:
                y = None

            v_pred = model(x=x_t, t=t, y=y)      # 无条件生成
            loss = rf.mse_loss(v_pred, x_1, x_0)

            loss.backward()
            optimizer.step()

            if batch % batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')

            loss_list.append(loss.item())

        scheduler.step()

        if epoch % checkpoint_save_interval == 0 or epoch == epochs - 1 or epoch == 0:
            # 第一轮也保存一下，快速测试用，大家可以删除
            # 保存模型
            print(f'Saving model {epoch} to {save_path}...')
            save_dict = dict(model=model.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=epoch,
                             loss_list=loss_list)
            torch.save(save_dict,
                       os.path.join(save_path, f'miniunet_{epoch}.pth'))


if __name__ == '__main__':
    train(config='./config/train_reflow_config.yaml')