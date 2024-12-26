import os
import scipy.io
import shutil
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import models
from torch import nn
import torch.nn.functional as NF
from torch.nn import DataParallel
from refer.utils import *
from refer.mymodels import Maxvit
import refer.img_processor as img_processor
from refer.Dataset import KinshipDataset, FIWDataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# 超参数设置
batch_size = 32
learning_rate = 1e-05
betas = (0.9, 0.999)
num_workers = 4
weight_decay = 1e-5  # 3e-05
pin_memory = True
drop_path_rate = 0.625
gamma = 0.7
num_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 数据集初始化
img_folder1 = '../data/KinFaceW-I/images'
img_folder2 = '../data/KinFaceW-II/images'
dataset1 = KinshipDataset(root_dirs=[img_folder1, img_folder2], transform=img_processor.preprocess224_MaxVit)
dataset2 = FIWDataset(csv_file="data/pair.csv", transform=img_processor.preprocess224_MaxVit)

# 合并原始数据集和新增数据集
dataset = ConcatDataset([dataset1, dataset2])
print("Total dataset length:", len(dataset))

# 划分训练集和测试集
dataset = torch.utils.data.Subset(dataset, range(0, 100000))
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

print("Train dataset length:", len(train_dataset))
print("Test dataset length:", len(test_dataset))


# 训练函数
def train(model, train_loader, val_loader, optimizer, num_epochs, batch_size, device, model_save_path, resume=False, model_name=None):
    print("Start training")
    writer = SummaryWriter()  # 初始化 TensorBoard
    model = DataParallel(model, device_ids=[0, 1, 2, 3])  # 使用多GPU
    model = model.to(device)

    best_accuracy = 0.0
    acc_list = []
    start_epoch = 0

    # 如果选择恢复训练
    if resume:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        acc_list = checkpoint['accuracy_list']
        print(f"Model loaded from {model_name} with best accuracy: {best_accuracy:.4f}")

    for epoch in range(start_epoch, num_epochs):
        model.train()  # 设置为训练模式
        train_correct = 0
        train_total = 0
        running_loss = 0.0

        # 遍历每个 batch
        for step, (imgs, label) in enumerate(train_loader, 1):
            imgs = (imgs[0].to(device), imgs[1].to(device))
            label = label.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = NF.cross_entropy(output, label.long())
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(output, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()
            running_loss += loss.item()

            # Step-level 监视
            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_loader) + step)
            writer.add_scalar('Accuracy/train_step', (predicted == label).sum().item() / label.size(0), epoch * len(train_loader) + step)

        # 计算每轮训练准确率
        train_accuracy = train_correct / train_total
        writer.add_scalar('Loss/train_epoch', running_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train_epoch', train_accuracy, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")

        # 验证集评估
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, label in val_loader:
                imgs = (imgs[0].to(device), imgs[1].to(device))
                label = label.to(device)
                output = model(imgs)
                loss = NF.cross_entropy(output, label.long())

                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_accuracy = val_correct / val_total
        writer.add_scalar('Loss/val_epoch', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/val_epoch', val_accuracy, epoch)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # 如果验证准确率提升，则保存模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            state_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'accuracy_list': acc_list,
            }
            save_path = f"{model_save_path}maxvit-{epoch}-{val_accuracy:.4f}.pth"
            torch.save(state_dict, save_path)
            print(f"Model saved with accuracy: {best_accuracy:.4f} at path: {save_path}")

        acc_list.append(val_accuracy)

    writer.close()


# 主函数
def main():
    model_save_path = 'model/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print("Model folder created")
    else:
        print("Model folder exists")

    # 初始化模型和优化器
    model = Maxvit(num_classes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    # 开始训练
    # 如果需要恢复训练，取消注释以下行
    # train(model, train_loader, test_loader, optimizer, num_epochs, batch_size, device, model_save_path, resume=True, model_name="path/to/checkpoint.pth")
    train(model, train_loader, test_loader, optimizer, num_epochs, batch_size, device, model_save_path)


if __name__ == "__main__":
    np.random.seed(0)
    main()
