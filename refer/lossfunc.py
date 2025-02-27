import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
import torch
import torch.nn as nn
import torch.nn.functional as F



class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.s = s  # Scaling factor
        self.m = m  # Angular margin
        self.easy_margin = easy_margin

        # Initialize weights for each class
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = self.sin_m * m

    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)  # Shape: (batch_size, feature_dim)
        weight = F.normalize(self.weight, p=2, dim=1)  # Shape: (num_classes, feature_dim)

        # Compute cosine similarity between features and weights
        cosine = F.linear(features, weight)  # Shape: (batch_size, num_classes)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))  # Shape: (batch_size, num_classes)
        phi = cosine * self.cos_m - sine * self.sin_m  # Shape: (batch_size, num_classes)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Combine phi and cosine based on one-hot labels
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # Apply scaling

        loss = F.cross_entropy(output, labels)
        return loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算欧氏距离
        pos_dist = F.pairwise_distance(anchor, positive, p=2)  # 锚点和正样本的距离
        neg_dist = F.pairwise_distance(anchor, negative, p=2)  # 锚点和负样本的距离

        # Triplet Loss 计算
        loss = torch.relu(pos_dist - neg_dist + self.margin)  # max(0, pos_dist - neg_dist + margin)
        return loss.mean()
class CosFaceLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, s=30.0, m=0.35):
        super(CosFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.s = s  # Scaling factor
        self.m = m  # Cosine margin

        # Initialize weights for each class
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)  # Shape: (batch_size, feature_dim)
        weight = F.normalize(self.weight, p=2, dim=1)  # Shape: (num_classes, feature_dim)

        # Compute cosine similarity between features and weights
        cosine = F.linear(features, weight)  # Shape: (batch_size, num_classes)
        phi = cosine - self.m  # Subtract margin

        # Create one-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Combine phi and cosine based on one-hot labels
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # Apply scaling

        loss = F.cross_entropy(output, labels)
        return loss
