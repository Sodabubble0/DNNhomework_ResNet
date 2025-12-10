import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json

# 【注意】导入新的 SE 模型
from models.se_resnet import se_resnet18
from utils.dataset import get_cifar10_loaders

# 配置参数
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 30
MILESTONES = [15, 25]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【注意】保存路径
SAVE_DIR = "./result/checkpoints"
LOG_DIR = "./result/logs"
LOG_FILENAME = 'se_resnet_history.json'

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

    return running_loss / len(loader), 100. * correct / total

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
    return test_loss / len(loader), 100. * correct / total

def main():
    print(f"开始训练 SE-ResNet-18 (通道注意力机制实验)...")
    print(f"Using Device: {DEVICE}")
    
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
    
    # 实例化 SE-ResNet-18
    model = se_resnet18(num_classes=10).to(DEVICE)
    
    # 计算增加的参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"SE-ResNet-18 参数量: {params/1e6:.2f} M (比原版略微增加)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, 
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)
    
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    
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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'se_resnet_best.pth'))
            
    with open(os.path.join(LOG_DIR, LOG_FILENAME), 'w') as f:
        json.dump(history, f)
        
    print(f"\n实验 C 完成！SE-ResNet 最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()