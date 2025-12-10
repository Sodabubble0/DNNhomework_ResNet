import matplotlib.pyplot as plt
import json
import os

# --------------------------------------------------------------------------
# 配置路径
# --------------------------------------------------------------------------
RESNET_LOG = "./result/logs/training_history.json"
PLAIN_LOG = "./result/logs/plain_cnn_history.json"
SAVE_DIR = "./result/figures"

os.makedirs(SAVE_DIR, exist_ok=True)

def plot_comparison():
    print("正在加载实验数据...")
    
    # 1. 检查文件是否存在
    if not os.path.exists(RESNET_LOG) or not os.path.exists(PLAIN_LOG):
        print("❌ 错误: 找不到日志文件。请确保 train.py 和 train_plain.py 都已运行完毕。")
        return

    # 2. 读取数据
    with open(RESNET_LOG, 'r') as f:
        res_hist = json.load(f)
    with open(PLAIN_LOG, 'r') as f:
        plain_hist = json.load(f)

    # 假设 Epochs 数量一致，取 ResNet 的长度
    epochs = range(1, len(res_hist['test_acc']) + 1)

    # 3. 创建画布
    plt.figure(figsize=(14, 6))

    # --- 左图：准确率对比 (Test Accuracy) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, res_hist['test_acc'], 'r-', linewidth=2, label='ResNet-18 (With Shortcut)')
    plt.plot(epochs, plain_hist['test_acc'], 'b--', linewidth=2, label='Plain CNN (No Shortcut)')
    
    plt.title('Experiment A: Test Accuracy Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 标注最终精度
    final_res_acc = res_hist['test_acc'][-1]
    final_plain_acc = plain_hist['test_acc'][-1]
    plt.text(len(epochs)-5, final_res_acc-2, f"{final_res_acc:.1f}%", color='red', fontweight='bold')
    plt.text(len(epochs)-5, final_plain_acc-2, f"{final_plain_acc:.1f}%", color='blue', fontweight='bold')

    # --- 右图：损失对比 (Test Loss) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, res_hist['test_loss'], 'r-', linewidth=2, label='ResNet-18')
    plt.plot(epochs, plain_hist['test_loss'], 'b--', linewidth=2, label='Plain CNN')
    
    plt.title('Experiment A: Test Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 4. 保存与显示
    save_path = os.path.join(SAVE_DIR, 'experiment_A_comparison.png')
    plt.savefig(save_path, dpi=300) # dpi=300 保证图片高清，适合插入 Word 报告
    print(f"✅ 对比图表已保存至: {save_path}")
    print("\n结论预判:")
    print(f"ResNet 最终精度: {final_res_acc:.2f}%")
    print(f"Plain  最终精度: {final_plain_acc:.2f}%")
    print(f"提升幅度: +{final_res_acc - final_plain_acc:.2f}%")
    
    plt.show()

if __name__ == '__main__':
    plot_comparison()