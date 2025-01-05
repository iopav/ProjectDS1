import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from refer.Dataset import KinshipDataset, FIWDataset
import refer.img_processor as img_processor
from refer.mymodels import Maxvit
from torch.utils.data import ConcatDataset, Subset, random_split

import numpy as np
np.random.seed(0)
# torch.manual_seed(0)
# 配置参数
img_folder1 = '../data/KinFaceW-I/images'
img_folder2 = '../data/KinFaceW-II/images'
fiw_csv = 'data/pair.csv'
batch_size = 16

print("loading data")

# 导入数据集
kinship_dataset = KinshipDataset(root_dirs=[img_folder1, img_folder2], transform=img_processor.preprocess224_MaxVit)
fiw_dataset = FIWDataset(csv_file=fiw_csv, transform=img_processor.preprocess224_MaxVit)
dataset = ConcatDataset([kinship_dataset, fiw_dataset])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机选择子集以减少内存占用
dataset_size = len(dataset)
print("dataset size:",dataset_size)
indices = np.random.permutation(dataset_size)


# # 数据集类型分布统计
# print("数据集类型分布")
# label_counts = torch.zeros(8, device=device)  # 假设有 8 类
# data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

# for data in data_loader:
#     imgs, labels = data
#     imgs = (imgs[0].to(device), imgs[1].to(device))  # 图像放入 GPU
#     labels = labels.to(device)  # 标签放入 GPU
#     for label in labels:
#         label_counts[label] += 1

# label_counts = label_counts.cpu().numpy()  # 转回 CPU 以便绘图
# print(f"Label counts: {dict(enumerate(label_counts.astype(int)))}")

# # 绘制数据分布图
# plt.figure(figsize=(10, 8))
# plt.pie(label_counts, labels=[f'Class {k}' for k in range(len(label_counts))],
#         autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
# plt.title('Dataset Type Distribution')
# plt.savefig('dataset_distribution.png')
# print("保存数据分布图为 'dataset_distribution.png'")
# plt.close()
# print("loading data done")

# 加载模型
model = Maxvit(num_classes=8)
checkpoint_path='model/maxvit-73-0.8542.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']

# Handle the 'module.' prefix issue
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_state_dict[key[7:]] = value  # Remove 'module.' prefix
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict)
# model.load_state_dict(torch.load('model/final/maxvit-262-0.8942.pth', map_location=device)['model_state_dict'])
model.to(device)
model.eval()
print("模型加载完成")

# 划分数据集

#从npy读入test数据集index
test_indices = np.load('test_indices.npy')
test_dataset = Subset(dataset, test_indices)
# 创建测试加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# 计算每种类型的准确率
correct = torch.zeros(8, device=device)  # 假设有 8 类
total = torch.zeros(8, device=device)

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = (imgs[0].to(device), imgs[1].to(device))
        labels = labels.to(device)

        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        for label, pred in zip(labels, preds):
            if label == pred:
                correct[label] += 1
            total[label] += 1

# 计算准确率
accuracies = (correct / total).cpu().numpy()  # 转回 CPU 以便绘图
categories = [f'Class {label}' for label in range(len(accuracies))]
accuracy_values = [acc if not np.isnan(acc) else 0 for acc in accuracies]  # 防止除零导致 NaN
print(f"准确率: {dict(zip(categories, accuracy_values))}")
# 绘制准确率图
plt.figure(figsize=(10, 6))
plt.bar(categories, accuracy_values, color='skyblue')
plt.ylim(0, 1)
plt.title('Model Accuracy by Category', fontsize=16, pad=20)  # 增加pad参数
plt.xlabel('Categories', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
for i, v in enumerate(accuracy_values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('model_accuracy.png')
print("保存分类准确率图为 'model_accuracy.png'")
plt.close()
