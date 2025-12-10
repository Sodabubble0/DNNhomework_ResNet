import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cv2  # 如果没有安装 opencv，请运行 pip install opencv-python

from models.resnet import resnet18
from utils.dataset import get_cifar10_loaders

# --------------------------------------------------------------------------
# 配置参数
# --------------------------------------------------------------------------
LOG_PATH = "./result/logs/training_history.json"
MODEL_PATH = "./result/checkpoints/resnet18_cifar10_best.pth"
SAVE_DIR = "./result/figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 功能 1: 绘制训练曲线 (Loss & Accuracy)
# --------------------------------------------------------------------------
def plot_curves():
    print("正在绘制训练曲线...")
    if not os.path.exists(LOG_PATH):
        print(f"❌ 错误: 找不到日志文件 {LOG_PATH}，请先运行 train.py")
        return

    with open(LOG_PATH, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 1. Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'r--', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['test_acc'], 'r--', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(SAVE_DIR, 'training_curves.png')
    plt.savefig(save_path)
    print(f"✅ 训练曲线已保存至: {save_path}")
    plt.show()

# --------------------------------------------------------------------------
# 功能 2: Grad-CAM (类激活映射) - 核心加分项
# --------------------------------------------------------------------------
class GradCAM:
    """
    简易版 Grad-CAM 实现
    原理: 捕获最后一层卷积的特征图(Feature Map)和梯度(Gradient)，
    计算权重后叠加回原图，显示模型关注区域。
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子 (Hooks)
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] 是梯度的 Tensor
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. 前向传播
        self.model.zero_grad()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # 2. 反向传播
        # 创建一个 target (one-hot)，只有目标类别为 1，其余为 0
        target = output[0][class_idx]
        target.backward()

        # 3. 生成 Heatmap
        # gradients: [1, 512, 1, 1] (对于 ResNet最后一层)
        # activations: [1, 512, 1, 1]
        
        # Global Average Pooling on Gradients (获取通道权重)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 将权重作用于特征图
        activation = self.activations[0] # [512, 1, 1]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        # 在通道维度求和，得到 2D Heatmap
        heatmap = torch.mean(activation, dim=0).cpu().detach()
        
        # ReLU + Normalize
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.numpy()

def visualize_gradcam():
    print("\n正在生成 Grad-CAM 热力图...")
    
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    model = resnet18(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 获取一张测试图片
    _, _, test_loader = get_cifar10_loaders(batch_size=1)
    img_tensor, label = next(iter(test_loader)) # 获取第一张图
    img_tensor = img_tensor.to(DEVICE)
    
    # 3. 初始化 Grad-CAM
    # ResNet 的最后一层卷积通常在 layer4 的最后一个 block
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    # 4. 运行 Grad-CAM
    heatmap = cam(img_tensor)

    # 5. 图像后处理 (叠加显示)
    # 反归一化原图用于显示
    norm_mean = np.array([0.4914, 0.4822, 0.4465])
    norm_std = np.array([0.2023, 0.1994, 0.2010])
    
    img_display = img_tensor.cpu().squeeze().numpy().transpose((1, 2, 0))
    img_display = img_display * norm_std + norm_mean
    img_display = np.clip(img_display, 0, 1)

    # 调整 Heatmap 大小与原图一致
    heatmap = cv2.resize(heatmap, (32, 32))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 叠加
    img_display_uint8 = np.uint8(255 * img_display)
    superimposed_img = cv2.addWeighted(img_display_uint8, 0.6, heatmap, 0.4, 0)
    # BGR 转 RGB
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # 6. 绘图
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    pred_class = classes[torch.argmax(model(img_tensor))]
    true_class = classes[label]

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_display)
    plt.title(f"Original: {true_class}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM (Pred: {pred_class})")
    plt.axis('off')

    save_path = os.path.join(SAVE_DIR, 'grad_cam_result.png')
    plt.savefig(save_path)
    print(f"✅ Grad-CAM 结果已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    # 确保有 opencv
    try:
        import cv2
    except ImportError:
        print("缺少 opencv-python 库，正在尝试安装...")
        os.system("pip install opencv-python")
        import cv2

    plot_curves()
    visualize_gradcam()