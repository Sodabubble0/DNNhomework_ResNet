import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNet-18/34 中使用的基础残差块 (Basic Block)
    结构: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Shortcut -> ReLU
    """
    expansion = 1 # 输出通道数相对于输入通道数的倍数 (BasicBlock为1, Bottleneck为4)

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 第一层卷积：可能进行下采样 (stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二层卷积：保持特征图大小不变 (stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ---------------------------------------------------------
        # Shortcut (捷径连接) 处理逻辑
        # 如果输入x和输出维度不同 (stride!=1 或 in_channels!=out_channels)，
        # 需要用 1x1 卷积调整 x 的维度，使其能与 F(x) 相加
        # ---------------------------------------------------------
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        # 1. 主路径 (F(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 2. 捷径连接 (x 或 1x1 conv(x))
        identity = self.shortcut(x)

        # 3. 核心步骤：残差相加
        out += identity
        
        # 4. 最后的激活
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # ---------------------------------------------------------
        # Stage 1: 初始处理层
        # 标准 ResNet 结构：7x7 Conv -> BN -> ReLU -> MaxPool
        # 注意：对于 CIFAR-10 (32x32)，这步下采样会比较激进，但为了符合任务书要求保留标准结构
        # ---------------------------------------------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---------------------------------------------------------
        # ResNet 核心层 (由多个 Block 堆叠而成)
        # ResNet-18 配置: [2, 2, 2, 2]
        # ---------------------------------------------------------
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # ---------------------------------------------------------
        # Head: 分类头
        # ---------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出变为 (B, C, 1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化 (Kaiming Normal) - 任务书要求
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride):
        """
        构建一个 Stage (包含多个 BasicBlock)
        只有第一个 Block 可能需要 stride!=1 进行下采样
        """
        layers = []
        # 第一个 block，处理 stride 和通道数变化
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        
        # 后续 block，stride 均为 1
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal (He init) - 针对 ReLU 优化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 nn.init.normal_(m.weight, 0, 0.01)
                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stage 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Main Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平为 (Batch, 512)
        x = self.fc(x)

        return x

def resnet18(num_classes=10):
    """
    构建 ResNet-18 模型
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=10):
    """
    构建 ResNet-34 模型
    结构配置: [3, 4, 6, 3]
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

# ---------------------------------------------------------
# 验收测试代码
# ---------------------------------------------------------
if __name__ == '__main__':
    # 1. 实例化模型
    model = resnet18()
    
    # 2. 打印模型结构
    print(model) 

    # 3. 随机输入测试
    # 修改：Batch Size 改为 2，避免最后一层 feature map 为 1x1 时 BatchNorm 报错
    dummy_input = torch.randn(2, 3, 32, 32)
    
    # 前向传播
    output = model(dummy_input)
    
    print("\n-----------------------------")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}") # 应该输出 [2, 10]
    
    if output.shape == (2, 10):
        print("✅ 测试通过！模型结构构建正确，可以处理 CIFAR-10 数据。")
    else:
        print("❌ 测试失败，输出维度不正确。")