import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

def get_cifar10_loaders(data_dir='./data', batch_size=128, num_workers=2, val_split=0.1):
    """
    下载并准备 CIFAR-10 的 DataLoader (训练集、验证集、测试集)。
    
    Args:
        data_dir (str): 数据存储目录
        batch_size (int): 批大小
        num_workers (int): 数据加载线程数
        val_split (float): 验证集占训练集的比例 (0.0 ~ 1.0)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # -----------------------------------------------------------
    # 1. 定义数据增强与预处理 (Transformations)
    # -----------------------------------------------------------
    # CIFAR-10 的均值和标准差 (这是业界标准值)
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.2023, 0.1994, 0.2010)

    # 训练集：需要数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 随机裁剪，填充4像素
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ToTensor(),                      # 转为 Tensor (CHW格式)
        transforms.Normalize(norm_mean, norm_std)   # 标准化
    ])

    # 验证集/测试集：不需要数据增强，仅需标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # -----------------------------------------------------------
    # 2. 下载并加载数据集
    # -----------------------------------------------------------
    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(f"正在加载 CIFAR-10 数据集到 {data_dir} ...")
    
    # 加载完整的训练集 (50000张)
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    # 加载测试集 (10000张)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # -----------------------------------------------------------
    # 3. 划分训练集与验证集
    # -----------------------------------------------------------
    # 计算划分大小
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    # 使用 random_split 随机划分
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # 固定种子确保可复现
    )
    
    # 注意：这里验证集也继承了 train_transform (含数据增强)。
    # 在严格的科研实验中，验证集最好不要数据增强。
    # 但为了简化代码结构，这里暂且复用。若需严格处理，需重写 Dataset 类或加载两次数据。

    # -----------------------------------------------------------
    # 4. 创建 DataLoader
    # -----------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"数据加载完成:")
    print(f"- 训练集大小: {len(train_dataset)} (Batch数量: {len(train_loader)})")
    print(f"- 验证集大小: {len(val_dataset)} (Batch数量: {len(val_loader)})")
    print(f"- 测试集大小: {len(test_dataset)} (Batch数量: {len(test_loader)})")

    return train_loader, val_loader, test_loader

def imshow(img, title=None):
    """
    可视化辅助函数：将 Tensor 图片反归一化并显示
    """
    # 反归一化：img = img * std + mean
    norm_mean = np.array([0.4914, 0.4822, 0.4465])
    norm_std = np.array([0.2023, 0.1994, 0.2010])
    
    img = img.numpy().transpose((1, 2, 0)) # 从 (C, H, W) 转为 (H, W, C)
    img = img * norm_std + norm_mean       # 反标准差和均值
    img = np.clip(img, 0, 1)               # 限制在 [0, 1] 范围内
    
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

# -----------------------------------------------------------
# 测试代码 (当直接运行此文件时执行)
# -----------------------------------------------------------
if __name__ == '__main__':
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    
    # 验收标准 1：打印形状
    # 获取一个 Batch 的数据
    images, labels = next(iter(train_loader))
    print(f"\n验收测试 - Batch 数据形状: {images.shape}") # 应为 [128, 3, 32, 32]
    print(f"验收测试 - Batch 标签形状: {labels.shape}") # 应为 [128]

    # 验收标准 2：随机取一张图用 Matplotlib 展示
    # CIFAR-10 类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 创建画布
    plt.figure(figsize=(10, 4))
    
    # 显示前 5 张图
    for i in range(5):
        plt.subplot(1, 5, i+1)
        imshow(images[i], title=classes[labels[i]])
    
    print("\n已弹出图像窗口，请检查是否显示正常。")
    plt.show()