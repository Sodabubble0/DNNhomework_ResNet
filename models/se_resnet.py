import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet  # 复用之前的主体框架

class SELayer(nn.Module):
    """
    SE Block (Squeeze-and-Excitation)
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 1. Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Excitation: 两个全连接层 (利用 1x1 卷积代替全连接层以减少 reshape 操作)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x)
        # Excitation
        y = self.fc(y)
        # Scale: 将权重乘回原输入
        return x * y

class SEBasicBlock(nn.Module):
    """
    带 SE 模块的 BasicBlock
    结构: Conv -> BN -> ReLU -> Conv -> BN -> SE -> Add -> ReLU
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(SEBasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- 插入 SE 模块 ---
        # reduction=16 是论文推荐的默认值
        self.se = SELayer(out_channels, reduction=16)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # --- 核心差异：在残差相加之前，先过 SE 模块 ---
        out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

def se_resnet18(num_classes=10):
    """
    构建 SE-ResNet-18
    使用我们新定义的 SEBasicBlock，传入原有的 ResNet 框架
    """
    return ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)