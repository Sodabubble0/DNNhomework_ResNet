import matplotlib.pyplot as plt
import json
import os

BASE_LOG = "./result/logs/training_history.json"
SE_LOG = "./result/logs/se_resnet_history.json"
SAVE_DIR = "./result/figures"

def plot_se_comparison():
    if not os.path.exists(BASE_LOG) or not os.path.exists(SE_LOG):
        print("❌ 请确保 train.py 和 train_se.py 都已运行完毕。")
        return

    with open(BASE_LOG, 'r') as f: base = json.load(f)
    with open(SE_LOG, 'r') as f: se = json.load(f)

    epochs = range(1, len(base['test_acc']) + 1)

    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, base['test_acc'], 'b--', linewidth=2, label=f'ResNet-18 (Base): {max(base["test_acc"]):.2f}%')
    plt.plot(epochs, se['test_acc'], 'r-', linewidth=2, label=f'SE-ResNet-18 (Attention): {max(se["test_acc"]):.2f}%')
    
    plt.title('Experiment C: Impact of SE-Block (Attention)')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(SAVE_DIR, 'experiment_C_se.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ SE 对比图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_se_comparison()