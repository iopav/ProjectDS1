import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from refer.Dataset import KinshipDataset, FIWDataset
import refer.img_processor as img_processor
from refer.mymodels import Maxvit
from torch.utils.data import ConcatDataset, Subset
import numpy as np

# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)
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
print(f"Using device: {device}")

# 检查 GPU 数量
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
if num_gpus < 4:
    raise RuntimeError("This code requires at least 4 GPUs to run.")

# 数据集大小和索引
dataset_size = len(dataset)
print("dataset size:", dataset_size)

# 从 .npy 文件加载测试集索引
test_indices = np.load('test_indices.npy')
test_dataset = Subset(dataset, test_indices)

# 创建测试加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# 加载模型
model = Maxvit(num_classes=8)
checkpoint_path = 'model/maxvit-73-0.8542.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']

# 处理 'module.' 前缀问题
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_state_dict[key[7:]] = value  # 去除 'module.' 前缀
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict)

# 使用 DataParallel 分配到多个 GPU
model = torch.nn.DataParallel(model)  # 包裹模型
model.to(device)
model.eval()
print("模型加载完成")

# 计算每种类型的准确率
correct = torch.zeros(8, device=device)  # 假设有 8 类
total = torch.zeros(8, device=device)

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = (imgs[0].to(device), imgs[1].to(device))  # 图像放入 GPU
        labels = labels.to(device)  # 标签放入 GPU

        outputs = model(imgs)  # 自动分发到多个 GPU
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
plt.title('Model Accuracy by Category', fontsize=16, pad=20)
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
