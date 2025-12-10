import matplotlib.pyplot as plt
import json
import os

R18_LOG = "./result/logs/training_history.json"
R34_LOG = "./result/logs/resnet34_history.json"
SAVE_DIR = "./result/figures"

def plot_depth_comparison():
    if not os.path.exists(R18_LOG) or not os.path.exists(R34_LOG):
        print("❌ 请确保 train.py 和 train_resnet34.py 都已运行完毕。")
        return

    with open(R18_LOG, 'r') as f: r18 = json.load(f)
    with open(R34_LOG, 'r') as f: r34 = json.load(f)

    epochs = range(1, len(r18['test_acc']) + 1)

    plt.figure(figsize=(10, 6))
    
    # 绘制 Accuracy 对比
    plt.plot(epochs, r18['test_acc'], 'b-', linewidth=2, label=f'ResNet-18 (Max: {max(r18["test_acc"]):.2f}%)')
    plt.plot(epochs, r34['test_acc'], 'r--', linewidth=2, label=f'ResNet-34 (Max: {max(r34["test_acc"]):.2f}%)')
    
    plt.title('Experiment B: ResNet-18 vs ResNet-34 (Depth Impact)')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(SAVE_DIR, 'experiment_B_depth.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 深度对比图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_depth_comparison()