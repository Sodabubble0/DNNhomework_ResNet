import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json

# 导入我们可以自定义的模块
from models.resnet import resnet18
from utils.dataset import get_cifar10_loaders

# --------------------------------------------------------------------------
# 超参数设置 (按照任务书要求)
# --------------------------------------------------------------------------
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 30  # 建议至少跑 20-30 轮，ResNet 需要多一点时间收敛
MILESTONES = [15, 25] # 在第 15 和 25 epoch 降低学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./result/checkpoints"
LOG_DIR = "./result/logs"

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """
    训练一个 Epoch
    """
    model.train() # 切换到训练模式 (启用 BN 和 Dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # 1. 梯度清零
        optimizer.zero_grad()
        
        # 2. 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 3. 反向传播
        loss.backward()
        
        # 4. 更新参数
        optimizer.step()
        
        # 统计数据
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    end_time = time.time()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Time: {end_time-start_time:.1f}s")
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    """
    验证/测试模型
    """
    model.eval() # 切换到评估模式 (锁定 BN 状态, 禁用 Dropout)
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # 禁用梯度计算，节省显存
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    avg_loss = test_loss / len(loader)
    acc = 100. * correct / total
    
    print(f"    >>> Test Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
    return avg_loss, acc

def main():
    print(f"Using Device: {DEVICE}")
    
    # 1. 准备数据
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
    
    # 2. 构建模型
    model = resnet18(num_classes=10).to(DEVICE)
    
    # 3. 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 4. 定义优化器 (SGD + Momentum)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, 
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # 5. 定义学习率调度器 (MultiStepLR)
    # 在指定的 milestones 节点将学习率乘以 0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)
    
    # 记录训练历史，用于后续画图
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [],  "test_acc": []
    }
    
    best_acc = 0.0
    
    print("开始训练...")
    start_global = time.time()
    
    for epoch in range(EPOCHS):
        # 训练一轮
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # 验证一轮
        v_loss, v_acc = evaluate(model, test_loader, criterion)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"    Current LR: {current_lr}")

        # 记录数据
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["test_loss"].append(v_loss)
        history["test_acc"].append(v_acc)
        
        # 保存最佳模型
        if v_acc > best_acc:
            print(f"    🎉 New Best Acc: {v_acc:.2f}% (Saved)")
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'resnet18_cifar10_best.pth'))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'resnet18_cifar10_last.pth'))
    
    # 保存训练日志 (JSON格式，方便 visualize 读取)
    with open(os.path.join(LOG_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)
        
    print(f"\n训练结束！总耗时: {(time.time() - start_global)/60:.1f} min")
    print(f"最佳准确率: {best_acc:.2f}%")
    print(f"日志已保存至: {LOG_DIR}")

if __name__ == '__main__':
    main()