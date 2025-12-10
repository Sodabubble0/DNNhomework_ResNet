import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json

# 【注意】这里导入 resnet34
from models.resnet import resnet34
from utils.dataset import get_cifar10_loaders

# --------------------------------------------------------------------------
# 配置参数 (保持与其他实验一致，控制变量)
# --------------------------------------------------------------------------
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 30
MILESTONES = [15, 25]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【注意】文件名区别开
SAVE_DIR = "./result/checkpoints"
LOG_DIR = "./result/logs"
LOG_FILENAME = 'resnet34_history.json'

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
    print(f"开始训练 ResNet-34 (深度对比实验)...")
    print(f"Using Device: {DEVICE}")
    
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
    
    # 【注意】实例化 ResNet-34
    model = resnet34(num_classes=10).to(DEVICE)
    
    # 打印参数量对比 (可选，为了报告写数据)
    params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-34 参数量: {params/1e6:.2f} M (ResNet-18 约为 11.17 M)")

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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'resnet34_cifar10_best.pth'))
            
    with open(os.path.join(LOG_DIR, LOG_FILENAME), 'w') as f:
        json.dump(history, f)
        
    print(f"\n实验 B 完成！ResNet-34 最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()