#加载程序所需要的包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os

def rightness(predictions, labels):
    '''rightness 计算预测错误率的函数
    
    :param predictions 是模型给出的一组预测结果,batch_size行num_classes列的矩阵
    :param labels是数据中的正确答案
    :return 返回数值为（正确样例数，总样本数）
    '''
    #对于任意一行（一个样本）的输出值的第1个维度求最大，得到每一行最大元素的下标
    pred = torch.max(predictions.data, 1)[1]
    #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素

#从硬盘文件夹中加载图像数据集
#数据存储总路径
# data_dir = 'data'
data_dir = 'F:\\data\\hymenoptera_data'
#图像的大小为224×224像素
image_size = 224
#从data_dir/train加载文件
#加载的过程将会对图像进行如下图像增强操作：
#1. 随机从原始图像中切下来一块224×224大小的区域
#2. 随机水平翻转图像
#3. 将图像的色彩数值标准化
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                    transforms.Compose([
                                    transforms.RandomResizedCrop(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                                    ])
                                    )
#加载校验数据集，对每个加载的数据进行如下处理：
#1. 放大到256×256像素
#2. 从中心区域切割下224×224大小的图像区域
#3. 将图像的色彩数值标准化
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                    transforms.Compose([
                                        # transforms.Scale(256), # 此方法在新版torch中已废弃
                                        transforms.Resize(256),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
                                    )
#创建相应的数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 4, shuffle = True, num_workers=4)
#读取数据中的分类类别数
num_classes = len(train_dataset.classes)
'''模型迁移'''
net = models.resnet18(pretrained=True)
for param in net.parameters():      # 返回的网络中所有可训练参数的集合
    param.requires_grad = False     # 原始的ResNet中的所有参数都设置成不需要计算梯度的属性
num_ftrs = net.fc.in_features       # num_ftrs存储了ResNet18最后的全连接层的输入神经元个数
net.fc = nn.Linear(num_ftrs, 2)     # 将原有的两层全连接层替换成一个输出单元为2的全连接层
criterion = nn.CrossEntropyLoss()   # 使用交叉熵损失函数
optimizer = optim.SGD(net.fc.parameters(), lr = 0.001, momentum=0.9) # 优化器使用带动量的随机梯度下降

#建立布尔变量，判断是否可以用GPU
use_cuda = torch.cuda.is_available()
#如果可以用GPU，则设定Tensor的变量类型支持GPU
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

#如果存在GPU，就将网络加载到GPU上
net = net.cuda() if use_cuda else net
# #将数据复制出来，然后加载到GPU上
# data, target = data.clone().detach().requires_grad(True),target.clone().detach()
# if use_cuda:
#     data, target = data.cuda(), target.cuda()
# #待计算完成后，需将数据放回CPU
# loss = loss.cpu() if use_cuda else loss

record = [] #记录准确率等数值的容器
#开始训练循环
num_epochs = 20 #训练20个epoch
net.train(True) #给网络模型做标记，说明模型在训练集上训练
for epoch in range(num_epochs):
    train_rights = [] #记录训练数据集准确率的容器
    train_losses = [] #记录训练数据损失函数的容器
    for batch_idx, (data, target) in enumerate(train_loader): #针对容器中的每一个批进行循环
        data, target = data.clone().detach().requires_grad_(True), target.clone().detach() #data为图像，target为标签
        output = net(data) #完成一次预测
        loss = criterion(output, target) #计算误差
        optimizer.zero_grad() #清空梯度
        loss.backward() #反向传播
        optimizer.step() #一步随机梯度下降
        #计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
        right = rightness(output, target)
        train_rights.append(right) #将计算结果装到列表容器中
        train_losses.append(loss.data.numpy()) #将计算结果装到列表容器中