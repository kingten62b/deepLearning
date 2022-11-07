import numpy as np
import pandas as pd #读取csv文件的库
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
#让输出图形直接在Notebook中显示
# %matplotlib inline

data_path = 'D:/myproject/py/deepLearning/02/hour.csv' #读取数据到内存，rides为一个dataframe对象
rides = pd.read_csv(data_path)
rides.head() #输出部分数据
counts = rides['cnt'][:50] #截取数据
x = np.arange(len(counts)) #获取变量x
y = np.array(counts) #单车数量为y
plt.figure(figsize = (10, 7)) #设定绘图窗口大小
plt.plot(x, y, 'o-') #绘制原始数据
plt.xlabel('X') #更改坐标轴标注
plt.ylabel('Y') #更改坐标轴标注
plt.show()