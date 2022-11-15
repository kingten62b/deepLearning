import numpy as np
import pandas as pd #读取csv文件的库
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
#让输出图形直接在Notebook中显示
# %matplotlib inline

data_path = 'D:/myproject/py/deepLearning/03/hour.csv' #读取数据到内存，rides为一个dataframe对象
rides = pd.read_csv(data_path)
rides.head() #输出部分数据
counts = rides['cnt'][:50] #截取数据

# x = np.arange(len(counts)) #获取变量x
# y = np.array(counts) #单车数量为y
# plt.figure(figsize = (10, 7)) #设定绘图窗口大小
# plt.plot(x, y, 'o-') #绘制原始数据
# plt.xlabel('X') #更改坐标轴标注
# plt.ylabel('Y') #更改坐标轴标注
# plt.show()


#输入变量，1,2,3,...这样的一维数组
# x = torch.FloatTensor(np.arange(len(counts), dtype = float))
x = torch.FloatTensor(np.arange(len(counts), dtype = float) / len(counts))  #将输入数据的范围做归一化处理
#输出变量，它是从数据counts中读取的每一时刻的单车数，共50个数据点的一维数组，作为标准答案
y = torch.FloatTensor(np.array(counts, dtype = float))
sz = 10 #设置隐含层神经元的数量
#初始化输入层到隐含层的权重矩阵，它的尺寸是(1,10)
weights = torch.randn((1, sz), requires_grad = True)
#初始化隐含层节点的偏置向量，它是尺寸为10的一维向量
biases = torch.randn((sz), requires_grad = True)
#初始化从隐含层到输出层的权重矩阵，它的尺寸是(10,1)
weights2 = torch.randn((sz, 1), requires_grad = True)


learning_rate = 0.001 #设置学习率
losses = [] #该数组记录每一次迭代的损失函数值，以方便后续绘图
x = x.view(50,-1)
y = y.view(50,-1)
for i in range(100000):
    #从输入层到隐含层的计算
    hidden = x * weights + biases
    #此时，hidden变量的尺寸是：(50,10)，即50个数据点，10个隐含层神经元
    #将sigmoid函数作用在隐含层的每一个神经元上
    hidden = torch.sigmoid(hidden)
    #隐含层输出到输出层，计算得到最终预测
    predictions = hidden.mm(weights2)
    #此时，predictions的尺寸为：(50,1)，即50个数据点的预测数值
    #通过与数据中的标准答案y做比较，计算均方误差
    loss = torch.mean((predictions - y) ** 2)
    #此时，loss为一个标量，即一个数
    losses.append(loss.data.numpy())
    if i % 10000 == 0: #每隔10000个周期打印一下损失函数数值
        print('loss:', loss)
    #*****************************************
    #接下来开始梯度下降算法，将误差反向传播
    loss.backward() #对损失函数进行梯度反传
    #利用上一步计算中得到的weights，biases等梯度信息更新weights或biases的数值
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    #清空所有变量的梯度值
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()


plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')