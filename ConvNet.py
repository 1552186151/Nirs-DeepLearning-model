# -*- coding: utf-8 -*- #
"""
@Project    ：NIR-Mathematical-Modeling-Tool
@File       ：standardization.py
@Author     ：ZAY
@Time       ：2023/3/22 17:21
@Annotation : "CNN模型定义 "
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable


# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

# in_channels (int) – 输入通道个数。在文本应用中，即为词向量的维度
# out_channels (int) – 输出通道个数 。有多少个out_channels，就需要多少个一维卷积（也就是卷积核的数量）
# kernel_size(int or tuple) – 卷积核的尺寸；卷积核的第二个维度由in_channels决定，所以实际上卷积核的大小为kernel_size * in_channels
# stride (int or tuple, optional) – 卷积操作的步长。 默认：1
# padding (int or tuple, optional) – 输入数据各维度各边上要补齐0的层数。 默认： 0
# dilation (int or tuple, optional) – 卷积核各元素之间的距离(空洞卷积大小)。 默认： 1
# groups (int, optional) – 输入通道与输出通道之间相互隔离的连接的个数(分组卷积设置)。 默认：1
# bias (bool, optional) – 如果被置为True，向输出增加一个偏差量，此偏差是可学习参数。 默认：True

# torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# num_features – 特征维度
# 当input的维度为（N, C）时，BN将对C维归一化；当input的维度为(N, C, L) 时，归一化的维度同样为C维。

# torch.nn.ReLU()
# y = 0, x <= 0; y = x, x > 0

# 计算输出特征图的大小：Outsize = (Insize - Kernelsize)/stride + 1 MaxPool padding = 0
# 计算一维卷积输出数量：out = (in + 2 * padding - Kernelsize)/stride + 1
# 输入通道可以理解为一个卷积核拥有几个通道
# 输出通道可以理解为拥有多少个卷积核

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size = 7, padding = 0),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size = 5, padding = 0),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size = 3, padding = 0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.AvgMaxPool = nn.AdaptiveMaxPool1d(output_size = 128,return_indices = False) # 70 140 280 自适应池化层 无论输入Input的size是多少，输出的size总为指定的size
        # self.AvgPool = nn.AdaptiveAvgPool1d(output_size = 128)
        self.fc = nn.Linear(8192, 1)  # 4480 8960 17920 全连接层
        self.drop = nn.Dropout(0.05)  # 每个神经元都有p概率的可能性置零

    def forward(self, out):
        out = self.conv1(out)
        # out = self.drop(out)
        out = self.conv2(out)
        out = self.conv3(out)
        # print(out.shape) # [16, 64, 2057]
        out = F.relu(self.AvgMaxPool(out))  # 自适应池化层 无论输入的size是多少，输出的size总为指定的size
        out = out.view(out.size(0), -1) # 根据原tensor数据和batchsize自动分配列数 类似于keras中的Flatten函数 多维的的数据平铺为一维
        out = self.fc(out)
        return out

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,out_C): # (8, 16, 16, 16, 48)
        super(Inception,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c, c1,kernel_size=1,padding=0),
            nn.Conv1d(c1, c1, kernel_size=3, padding=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c, c2,kernel_size=1,padding=0),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2)
        )
        self.p3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c, c3,kernel_size=3,padding=1),
        )
        self.conv_linear = nn.Conv1d((c1+c2+c3), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),

            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        out =  torch.cat((p1,p2,p3),dim=1)
        out += self.short_cut(x)
        return out

class DeepSpectra(nn.Module):
    def __init__(self):
        super(DeepSpectra, self).__init__()
        self.conv1 = nn.Sequential(
            # nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=0)
            # nn.Conv1d(1, 8, kernel_size = 5, stride = 3, padding = 0)
            nn.Conv1d(1, 4, kernel_size = 5, stride = 3, padding = 0)
        )
        # self.Inception = Inception(16, 32, 32, 32, 96)
        # self.Inception = Inception(8, 16, 16, 16, 48)
        self.Inception = Inception(4, 8, 8, 8, 24)
        self.fc = nn.Sequential(
            nn.Linear(16536, 1),
            # nn.Dropout(0.5),
            # nn.Linear(5000, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Inception(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x