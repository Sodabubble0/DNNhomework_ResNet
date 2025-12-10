import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json

# 【注意】这里导入的是我们刚写的 Plain CNN
from models.plain_cnn import plain_cnn18
from utils.dataset import get_cifar10_loaders

# 超参数 (保持与 ResNet 完全一致，控制变量)
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 30
MILESTONES = [15, 25]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【注意】修改保存路径，避免覆盖 ResNet 的结果
SAVE_DIR = "./result/checkpoints"
LOG_DIR = "./result/logs"
LOG_FILENAME = 'plain_cnn_history.json' # 不同的日志名

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
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
    return avg_loss, acc

def main():
    print(f"开始训练 Plain CNN (无残差结构) 对比实验...")
    print(f"Using Device: {DEVICE}")
    
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
    
    # 实例化 Plain CNN
    model = plain_cnn18(num_classes=10).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, 
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)
    
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    
    start_global = time.time()
    
    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {t_acc:.2f}% | Test Acc: {v_acc:.2f}%")

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["test_loss"].append(v_loss)
        history["test_acc"].append(v_acc)
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'plain_cnn_best.pth'))
    
    # 保存 Plain CNN 的日志
    with open(os.path.join(LOG_DIR, LOG_FILENAME), 'w') as f:
        json.dump(history, f)
        
    print(f"\n实验 A 完成！最佳准确率: {best_acc:.2f}%")
    print(f"日志已保存至: {os.path.join(LOG_DIR, LOG_FILENAME)}")

if __name__ == '__main__':
    main()