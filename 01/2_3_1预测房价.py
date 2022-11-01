import torch
import matplotlib.pyplot as plt #导入画图的程序包

#生成数据并添加噪声
x = torch.linspace(0, 100, 100).type(torch.FloatTensor)
rand =torch.randn(100)* 10
y = x + rand
#划分测试与训练数据
x_train = x[: -10]
x_test = x[-10 :]
y_train = y[: -10]
y_test = y[-10 :]

plt.figure(figsize=(10,8)) #设定绘制窗口大小为10*8 inch
#绘制数据，由于x和y都是自动微分变量，需要用data获取它们包裹的Tensor，并转成Numpy plt.plot(x_train.data.n
plt.xlabel('X') #添加X轴的标注
plt.ylabel('Y') #添加Y轴的标注
plt.show() #画出图形