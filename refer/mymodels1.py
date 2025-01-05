

import torch
import torch.nn as nn
from torchvision import models

# 1. 定义 EfficientNet 模型，并保留特征向量输出，同时加一个线性分类层

class EfficientNetWithHead(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetWithHead, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()  
        self.feature_dim = None
        self._infer_feature_dim()

        self.fc_1 = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, 512))    
        self.fc_2 = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, 512))
        
        self.fc_3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))
    def _infer_feature_dim(self):
        # 利用注册钩子获取特征维度
        def hook(module, input, output):
            self.feature_dim = output.shape[1]
        
        handle = self.backbone.register_forward_hook(hook)
        with torch.no_grad():
            # 随机传入一个输入，触发钩子函数
            dummy_input = torch.randn(1, 3, 224, 224)  
            self.backbone(dummy_input)
        handle.remove()  # 移除钩子以防干扰    
        
    def forward(self, x):
        feature1 = self.backbone(x[0])  # 提取特征向量
        feature2 = self.backbone(x[1])
        t_1 = self.fc_1(feature1)
        t_2 = self.fc_2(feature2)
        #求出 t1转变为t2的转换矩阵x t1*x = t2
        x=torch.matmul(t_1.t(),t_2)
        
        
        # 分类器，将特征映射为类别输出
        return self.fc_3(x, dim=1)







class Maxvit(nn.Module):
    def __init__(self, num_classes=8):
        super(Maxvit, self).__init__()
        
        # 加载 MaxVit 主干网络
        self.backbone = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()  # 移除分类器
        
        # 初始化特征维度
        self.feature_dim = None
        self._infer_feature_dim()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        # 定义全连接层
        self.fc_1 = nn.Sequential(
            nn.Linear(self.feature_dim, 1280),
            nn.ReLU(),
            nn.Linear(1280, 512)
        )    
        
        self.fc_2 = nn.Sequential(
            nn.Linear(self.feature_dim, 1280),
            nn.ReLU(),
            nn.Linear(1280, 512)
        )
        
        # self.fc_3 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes)
        # )
        self.fc_3 = nn.Sequential(
            nn.Linear(512*512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )#x

    def _infer_feature_dim(self):
        """通过钩子获取主干网络的输出特征维度"""
        def hook(module, input, output):
            self.feature_dim = output.shape[1]
            # print(f'Feature dimension: {self.feature_dim}')
        
        # 注册钩子到主干网络最后一层
        handle = self.backbone.register_forward_hook(hook)
        
        # 传入一个随机输入以触发钩子并获取输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            self.backbone(dummy_input)
        
        # 移除钩子
        handle.remove()

    # def forward(self, x):
    #     """前向传播，处理两个输入图像"""
    #     feature1 = self.backbone(x[0])  # 图像 1 的特征
    #     feature2 = self.backbone(x[1])  # 图像 2 的特征
    #     feature1 = self.pooling(feature1).view(feature1.size(0), -1)
    #     feature2 = self.pooling(feature2).view(feature2.size(0), -1)
    #     t_1 = self.fc_1(feature1)  # 映射到嵌入空间
    #     t_2 = self.fc_2(feature2)
        
    #     # 将两个特征相加后通过最终分类器
    #     combined_features = t_1 + t_2
    #     #求出 t1转变为t2的转换矩阵x t1*x = t2
    #     # combined_features=torch.matmul(t_1.t(),t_2)
        
    #     output = self.fc_3(combined_features)
    
        
    #     return output
    def forward(self, x):#x
        """前向传播，处理两个输入图像"""
        # 获取 Backbone 的嵌入特征
        feature1 = self.backbone(x[0])  # 图像 1 的特征
        feature2 = self.backbone(x[1])  # 图像 2 的特征
        feature1 = self.pooling(feature1).view(feature1.size(0), -1)  # [batch_size, feature_dim]
        feature2 = self.pooling(feature2).view(feature2.size(0), -1)  # [batch_size, feature_dim]
        
        # 将特征映射到嵌入空间
        t_1 = self.fc_1(feature1)  # [batch_size, 512]
        t_2 = self.fc_2(feature2)  # [batch_size, 512]
        
        # 初始化一个变形矩阵列表
        transformation_matrices = []
        with torch.cuda.amp.autocast(enabled=False):
            for i in range(t_1.size(0)):  # 遍历每个样本
                t1_sample = t_1[i].unsqueeze(0)  # [1, 512]
                t2_sample = t_2[i].unsqueeze(0)  # [1, 512]
                #trans t1sample to float32
                t1_sample=t1_sample.float()
                t2_sample=t2_sample.float()
                
                # 计算变形矩阵 X: (t1^T * t1)^(-1) * t1^T * t2
                # pseudo_inverse = torch.linalg.pinv((t1_sample.T @ t1_sample).float())
                # pseudo_inverse = torch.linalg.pinv(t1_sample.T @ t1_sample)  # [512, 512]
                pseudo_inverse = torch.linalg.pinv((t1_sample.T @ t1_sample).float())

                transformation_matrix = pseudo_inverse @ t1_sample.T @ t2_sample  # [512, 512]
                transformation_matrices.append(transformation_matrix)
        
        # 堆叠所有变形矩阵并展平为向量
        transformation_matrices = torch.stack(transformation_matrices)  # [batch_size, 512, 512]
        transformation_vectors = transformation_matrices.view(t_1.size(0), -1)  # [batch_size, 512*512]
        
        # 输入到全连接层
        output = self.fc_3(transformation_vectors)
        
        return output

