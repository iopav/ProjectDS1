
#
#module load tensorboard/2.15.1-gfbf-2023a torchvision/0.16.0-foss-2023a-CUDA-12.1.1 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 virtualenv/20.23.1-GCCcore-12.3.0 matplotlib/3.7.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a h5py/3.9.0-foss-2023a JupyterLab/4.0.5-GCCcore-12.3.0
#source yz_venv/bin/activate
#impot
import os
import scipy.io
import shutil
import math
import os

#将merge的图片读入dataloader
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torch
from torchvision import models
from torch import nn
import torch.nn.functional as NF
# from refer.process import train_epoch
# from refer.models import get_maxvit
from refer.utils import *
from torch.nn import Parameter
from refer.mymodels import Maxvit
from torch.utils.data import random_split


import refer.img_processor as img_processor

import matplotlib.pyplot as plt
from PIL import Image
from refer.Dataset import KinshipDataset,FIWDataset
from torch.utils.tensorboard import SummaryWriter





#split and init
img_folder1 = '..\data\\KinFaceW-I\\images'
img_folder2 = '..\data\\KinFaceW-II\\images'
dataset = KinshipDataset(root_dirs=[img_folder1, img_folder2],transform=img_processor.preprocess224_MaxVit)
dataset1 = FIWDataset(csv_file="data/pair.csv",transform=img_processor.preprocess224_MaxVit)
# fiwdataset =
# 合并原始数据集和新增数据集
dataset = ConcatDataset([dataset, dataset1])
print("dataset len",dataset.__len__())
#parameters
batch_size = 8
learning_rate = 1e-05
betas = (0.9, 0.999)
num_workers = 4
weight_decay = 1e-5 #3e-05
pin_memory = True
drop_path_rate = 0.625
gamma = 0.7
num_epochs=200
torch.manual_seed(0)
# dataset = torch.utils.data.Subset(dataset, range(0, 45))
# Define the lengths for train, validation, and test sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = Maxvit(num_classes=8).to(device)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
print("train len, test len:",train_dataset.__len__(),  test_dataset.__len__())
# print("train shape:",train_dataset.__getitem__(0)[0][0].shape)


# Train the model
def train(model, train_loader, val_loader, optimizer, num_epochs, batch_size, device, model_save_path,resume=False,model_name=None):
    # Initialize SummaryWriter
    print("start training")    
    writer = SummaryWriter()
    best_accuracy = 0.0
    acc_list = []
    if resume:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        acc_list = checkpoint['accuracy_list']
        print(f'Model loaded from {model_save_path} with accuracy: {best_accuracy}')
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_correct = 0
        train_total = 0
        running_loss = 0.0

        # Training loop
        for imgs, label in train_loader:
            # print(f"img1 shape: {imgs[0].shape}, img2 shape: {imgs[1].shape}, label shape: {label.shape}")
    
            imgs = (imgs[0].to(device), imgs[1].to(device))
            label = label.to(device)
            # assert torch.max(label) < 8 and torch.min(label) >= 0, "Label out of range!"

            optimizer.zero_grad()
            output = model(imgs)
            loss = NF.cross_entropy(output, label.long())
            loss.backward()
            optimizer.step()

            # Calculate accuracy for this batch
            _, predicted = torch.max(output, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()
            running_loss += loss.item()
            # print(f'Loss: {loss.item()}')

        train_accuracy = train_correct / train_total
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}')

        # Validation
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
        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(f'Validation Accuracy: {val_accuracy:.4f}')

        # Save the model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            state_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'accuracy_list': acc_list,
            }
            torch.save(state_dict, model_save_path+'maxvit-'+str(epoch)+"-"+str(val_accuracy)+'.pth')
            print(f'Model saved with accuracy: {best_accuracy:.4f}',"at path:",model_save_path+'maxvit-'+str(epoch)+"-"+str(val_accuracy)+'.pth')

        acc_list.append(val_accuracy)

    writer.close()
    
    
    
def main():
    model_save_path = 'model/'
    #确保文件夹存在，否则创建
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print("model folder created")
    else:
        print("model folder exists")
    # train(model, train_loader, test_loader, optimizer, num_epochs, batch_size, device, model_save_path,resume=True,model_name='E:\workspace\ProjectDS\model\maxvit00.5424836601307189.pth')
    train(model, train_loader, test_loader, optimizer, num_epochs, batch_size, device, model_save_path)
if __name__ == "__main__":
    main()