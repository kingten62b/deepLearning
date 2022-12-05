from torch import nn
import torch.nn.functional as F
import torch
from config import IMG_SIZE

'''
定义卷积神经网络
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层激活->池化->Dropout
        # input 3*64*64-卷积->16*64*64-池化->16*32*32
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        # 第二层卷积->激活->池化->Dropout
        # input 16*32*32-卷积->32*32*32-池化->32*16*16
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        # 第三层卷积->激活->池化->Dropout
        # input 32*16*16-卷积->32*16*16-池化->32*8*8
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        # 全连接层 
        self.out = nn.Linear(32 * IMG_SIZE//8 * IMG_SIZE//8 , 4025)
        
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        # 对结果进行log + softmax并输出
        return F.log_softmax(x, dim=1)

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