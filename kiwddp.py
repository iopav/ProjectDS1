import os
import scipy.io
import shutil
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import models
from torch import nn
import torch.nn.functional as NF
from PIL import Image
import matplotlib.pyplot as plt
from refer.utils import *
from refer.mymodels1 import Maxvit
from refer.Dataset import KinshipDataset, FIWDataset
import refer.img_processor as img_processor
from torch.utils.tensorboard import SummaryWriter
# torch.cuda.set_max_split_size_mb(128)  # 调整此值减少显存碎片化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import numpy as np
from torch.utils.data import Subset
# Initialize distributed training
os.environ['MASTER_ADDR'] = 'localhost'  # Replace with the master node's IP if using multiple machines
os.environ['MASTER_PORT'] = '12355'     # A free port number for communication
torch.manual_seed(0)
# Parameters
batch_size = 16
learning_rate = 1e-05
betas = (0.9, 0.999)
num_workers = 4
weight_decay = 1e-5  # 3e-05
drop_path_rate = 0.625
gamma = 0.7
num_epochs = 500
model_save_path = 'model/'
     
def setup(rank, world_size):
    """Set up the process group for distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the process group after training."""
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Ensure model save path exists
    if rank == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            print("Model folder created")
        else:
            print("Model folder exists")

    # Dataset and DataLoader
    img_folder1 = '../data/KinFaceW-I/images'
    img_folder2 = '../data/KinFaceW-II/images'
    dataset = KinshipDataset(root_dirs=[img_folder1, img_folder2], transform=img_processor.preprocess224_MaxVit)
    dataset1 = FIWDataset(csv_file="data/pair.csv", transform=img_processor.preprocess224_MaxVit)
    dataset = ConcatDataset([dataset, dataset1])
    # Randomly shuffle the indices of the dataset
    dataset_size = len(dataset)
    indices = np.random.permutation(dataset_size)
    
    # Select the first 100,000 random indices
    # subset_indices = indices[100000:200000]
    
    # Create a subset dataset
    # dataset = Subset(dataset, subset_indices)
    # Split dataset into train and test
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #保存test数据集index
    test_indices = test_dataset.indices if isinstance(test_dataset, Subset) else list(range(len(test_dataset)))
    np.save('test_indices.npy', test_indices)

    
    # Use DistributedSampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    # Model, optimizer, and loss function
    device = torch.device('cuda', rank)
    model = Maxvit(num_classes=8).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    # Initialize automatic mixed precision (AMP)
    scaler = torch.cuda.amp.GradScaler()

    # TensorBoard writer (only for rank 0)
    writer = SummaryWriter() if rank == 0 else None

    # Training loop
    best_accuracy = 0.0
    acc_list = []
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        train_correct = 0
        train_total = 0
        running_loss = 0.0

        for step, (imgs, label) in enumerate(train_loader, 1):
            imgs = (imgs[0].to(device), imgs[1].to(device))
            label = label.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(imgs)
                loss = NF.cross_entropy(output, label.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(output, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()
            running_loss += loss.item()

            # Step-level monitoring only for rank 0
            if rank == 0:
                writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_loader) + step)
                writer.add_scalar('Accuracy/train_step', (predicted == label).sum().item() / label.size(0), epoch * len(train_loader) + step)

        train_accuracy = train_correct / train_total
        if rank == 0:
            writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}')
        scheduler.step()
        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, label in test_loader:
                imgs = (imgs[0].to(device), imgs[1].to(device))
                label = label.to(device)

                with torch.cuda.amp.autocast():
                    output = model(imgs)
                    loss = NF.cross_entropy(output, label.long())

                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_accuracy = val_correct / val_total
        if rank == 0:
            writer.add_scalar('Loss/val', val_loss / len(test_loader), epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            print(f'Validation Accuracy: {val_accuracy:.4f}')

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                state_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_accuracy': best_accuracy,
                    'accuracy_list': acc_list,
                }
                save_path = model_save_path + f'maxvit-{epoch}-{val_accuracy:.4f}.pth'
                torch.save(state_dict, save_path)
                print(f'Model saved with accuracy: {best_accuracy:.4f} at path: {save_path}')

            acc_list.append(val_accuracy)

    if rank == 0:
        writer.close()

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True,)

if __name__ == "__main__":
    np.random.seed(0)
    main()
