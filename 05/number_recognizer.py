#导入所需要的包，请保证torchvision已经在你的环境中安装好
#在Windows中，需要单独安装torchvision包，在命令行运行pip install torchvision即可
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#以下语句可以让Jupyter Notebook直接输出图像
%matplotlib inline


def rightness(predictions, labels):
    '''计算预测错误率的函数,其中predictions是模型给出的一组预测结果,batch_size行num_classes列的矩阵,labels是数据中的正确答案'''
    #对于任意一行（一个样本）的输出值的第1个维度求最大，得到每一行最大元素的下标
    pred = torch.max(predictions.data, 1)[1]
    #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素


"""5.2.1 数据准备"""
image_size = 28 #图像的总尺寸为28×28
num_classes = 10 #标签的种类数
num_epochs = 20 #训练的总循环周期
batch_size = 64 #一个批次的大小，64张图片

#加载MNIST数据，如果没有下载过，系统就会在当前路径下新建/data子目录，并把文件存放其中（压缩的格式）
#MNIST数据属于torchvision包自带的数据，可以直接调用
#当用户想调用自己的图像数据时，可以用torchvision.datasets.ImageFolder
#或torch.utils.data.TensorDataset来加载
train_dataset = dsets.MNIST(root='./data', #文件存放路径
                            train=True, #提取训练集
                            #将图像转化为Tensor，在加载数据时，就可以对图像做预处理
                            transform=transforms.ToTensor(),
                            download=True) #当找不到文件的时候，自动下载
#加载测试数据集
test_dataset = dsets.MNIST( root='./data',
                            train=False,
                            transform=transforms.ToTensor())
#训练数据集的加载器，自动将数据切分成批，顺序随机打乱
train_loader = torch.utils.data.DataLoader( dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

'''我们希望将测试数据分成两部分，一部分作为校验数据，一部分作为测试数据。
校验数据用于检测模型是否过拟合并调整参数，测试数据检验整个模型的工作'''

#首先，定义下标数组indices，它相当于对所有test_dataset中数据的编码
#然后，定义下标indices_val表示校验集数据的下标，indices_test表示测试集的下标
indices = range(len(test_dataset))
indices_val = indices[:5000]
indices_test = indices[5000:]
#根据下标构造两个数据集的SubsetRandomSampler采样器，它会对下标进行采样
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val) # 返回打乱的数据
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)
#根据两个采样器定义加载器
#注意将sampler_val和sampler_test分别赋值给了validation_loader和test_loader
validation_loader = torch.utils.data.DataLoader(dataset =test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                sampler = sampler_val
                                                )
test_loader = torch.utils.data.DataLoader(  dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle= False,
                                            sampler = sampler_test
                                            )
#随便从数据集中读入一张图片，并绘制出来
idx = 100
#dataset支持下标索引，其中提取出来的元素为features、target格式，即属性和标签。[0]表示索引features
muteimg = train_dataset[idx][0].numpy()
#一般的图像包含RGB这3个通道，而MNIST数据集的图像都是灰度的，只有一个通道
#因此，我们忽略通道，把图像看作一个灰度矩阵
#用imshow画图，会将灰度矩阵自动展现为彩色，不同灰度对应不同的颜色：从黄到紫
plt.imshow(muteimg[0,...])
plt.show()
print('标签是：',train_dataset[idx][1])


"""5.2.2 构建网络"""
#定义卷积神经网络：4和8为人为指定的两个卷积层的厚度（feature map的数量）
depth = [4, 8]
class ConvNet(nn.Module):
    def __init__(self):
        #该函数在创建一个ConvNet对象即调用语句net=ConvNet()时就会被调用
        #首先调用父类相应的构造函数
        super(ConvNet, self).__init__()
        #其次构造ConvNet需要用到的各个神经模块
        #注意，定义组件并不是真正搭建组件，只是把基本建筑砖块先找好
        #定义一个卷积层，输入通道为1，输出通道为4，窗口大小为5，padding为2
        self.conv1 = nn.Conv2d(1, 4, 5, padding = 2)
        self.pool = nn.MaxPool2d(2, 2) #定义一个池化层，一个窗口为2×2的池化运算
        #第二层卷积，输入通道为depth[0]，输出通道为depth[1]，窗口为5，padding为2
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding = 2)
        #一个线性连接层，输入尺寸为最后一层立方体的线性平铺，输出层512个节点
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1] , 512)
        self.fc2 = nn.Linear(512, num_classes) #最后一层线性分类单元，输入为512，输出为要做分类的类别数
    def forward(self, x): #该函数完成神经网络真正的前向运算，在这里把各个组件进行实际的拼装
        #x的尺寸：(batch_size, image_channels, image_width, image_height)
        x = self.conv1(x) #第一层卷积
        x = F.relu(x) #激活函数用ReLU，防止过拟合
        #x的尺寸：(batch_size, num_filters, image_width, image_height)
        x = self.pool(x) #第二层池化，将图片变小
        #x的尺寸：(batch_size, depth[0], image_width/2, image_height/2)
        x = self.conv2(x) #第三层又是卷积，窗口为5，输入输出通道分别为depth[0]=4, depth[1]=8
        x = F.relu(x) #非线性函数
        #x的尺寸：(batch_size, depth[1], image_width/2, image_height/2)
        x = self.pool(x) #第四层池化，将图片缩小到原来的1/4
        #x的尺寸：(batch_size, depth[1], image_width/4, image_height/4)
        #将立体的特征图tensor压成一个一维的向量
        #view函数可以将一个tensor按指定的方式重新排布
        #下面这个命令就是要让x按照batch_size * (image_size//4)^2*depth[1]的方式来排布向量
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        #x的尺寸：(batch_size, depth[1]*image_width/4*image_height/4)
        x = F.relu(self.fc1(x)) #第五层为全连接，ReLU激活函数
        #x的尺寸：(batch_size, 512)
        #以默认0.5的概率对这一层进行dropout操作，防止过拟合
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) #全连接
        #x的尺寸：(batch_size, num_classes)
        #输出层为log_softmax，即概率对数值log(p(x))。采用log_softmax可以使后面的交叉熵计算更快
        x = F.log_softmax(x, dim=1)
        return x
    def retrieve_features(self, x):
        #该函数用于提取卷积神经网络的特征图，返回feature_map1, feature_map2为前两层卷积层的特征图
        feature_map1 = F.relu(self.conv1(x)) #完成第一层卷积
        x = self.pool(feature_map1) #完成第一层池化
        #第二层卷积，两层特征图都存储到了feature_map1, feature_map2中
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1, feature_map2)


"""5.2.3 运行模型"""
net = ConvNet() #新建一个卷积神经网络的实例，此时ConvNet的__init__()函数会被自动调用
criterion = nn.CrossEntropyLoss() #Loss函数的定义，交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #定义优化器，普通的随机梯度下降算法
record = [] #记录准确率等数值的容器
weights = [] #每若干步就记录一次卷积核
#开始训练循环
for epoch in range(num_epochs):
    train_rights = [] #记录训练数据集准确率的容器
    ''' 下面的enumerate起到构造一个枚举器的作用。在对train_loader做循环迭代时，enumerate会自动输出一个数字指示循环了几次，并记录在batch_idx中，它就等于0，1，2，...train_loader每迭代一次，就会输出一对数据data和target，分别对应一个批中的手写数字图及对应的标签。'''
    for batch_idx, (data, target) in enumerate(train_loader): #针对容器中的每一个批进行循环
        data, target = data.clone().requires_grad_(True), target.clone().detach()
        #给网络模型做标记，标志着模型在训练集上训练
        #这种区分主要是为了打开关闭net的training标志，从而决定是否运行dropout
        net.train()
        output = net(data) #神经网络完成一次前馈的计算过程，得到预测输出output
        loss = criterion(output, target) #将output与标签target比较，计算误差
        optimizer.zero_grad() #清空梯度
        loss.backward() #反向传播
        optimizer.step() #一步随机梯度下降算法
        right = rightness(output, target) #计算准确率所需数值，返回数值为（正确样例数，总样本数）
        train_rights.append(right) #将计算结果装到列表容器train_rights中
        if batch_idx % 100 == 0: #每间隔100个batch执行一次打印操作
            net.eval() #给网络模型做标记，标志着模型在训练集上训练
            val_rights = [] #记录校验数据集准确率的容器
            #开始在校验集上做循环，计算校验集上的准确度
            for (data, target) in validation_loader:
                data, target = data.clone().requires_grad_(True), target.clone().detach()
                #完成一次前馈计算过程，得到目前训练得到的模型net在校验集上的表现
                output = net(data)
                #计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                right = rightness(output, target)
                val_rights.append(right)
        #分别计算目前已经计算过的测试集以及全部校验集上模型的表现：分类准确率
        #train_r为一个二元组，分别记录经历过的所有训练集中分类正确的数量和该集合中总的样本数
        #train_r[0]/train_r[1]是训练集的分类准确度，val_r[0]/val_r[1]是校验集的分类准确度
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        #val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
        val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
        #打印准确率等数值，其中正确率为本训练周期epoch开始后到目前批的正确率的平均值
        print('训练周期：{} [{}/{} ({:.0f}%)]\t，Loss：{:.6f}\t，训练正确率：{:.2f}%\t，校验正确率：{:.2f}%'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data,
            100. * train_r[0] / train_r[1],
            100. * val_r[0] / val_r[1]))
        #将准确率和权重等数值加载到容器中，方便后续处理
        record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))
        #weights记录了训练周期中所有卷积核的演化过程，net.conv1.weight提取出了第一层卷积核的权重
        #clone是将weight.data中的数据做一个备份放到列表中
        #否则当weight.data变化时，列表中的每一项数值也会联动
        #这里使用clone这个函数很重要
        weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
            net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])


"""5.2.4 测试模型"""
#在测试集上分批运行，并计算总的正确率
net.eval() #标志着模型当前为运行阶段
vals = [] #记录准确率所用列表
#对测试数据集进行循环
for data, target in test_loader:
    data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
    output = net(data) #将特征数据输入网络，得到分类的输出
    val = rightness(output, target) #获得正确样本数以及总样本数
    vals.append(val) #记录结果
#计算准确率
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0 * rights[0] / rights[1]
right_rate


#绘制训练过程的误差曲线，校验集和测试集上的错误率
plt.figure(figsize = (10, 7))
plt.plot(record) #record记载了每一个打印周期记录的训练集和校验集上的准确度
plt.xlabel('Steps')
plt.ylabel('Error rate')
plt.show()


"""5.3 剖析卷积神经网络"""
#提取第一层卷积层的卷积核
plt.figure(figsize = (10, 7))
for i in range(4):
    plt.subplot(1,4,i + 1)
    #提取第一层卷积核中的权重值，注意conv1是net的属性
    plt.imshow(net.conv1.weight.data.numpy()[i,0,...])
    # plt.show()

#调用net的retrieve_features方法可以抽取出输入当前数据后输出的所有特征图（第一个卷积层和第二个卷积层）
#首先定义读入的图片，它是从test_dataset中提取第idx个批次的第0个图
#其次unsqueeze的作用是在最前面添加一维
#目的是让这个input_x的tensor是四维的，这样才能输入给net。补充的那一维表示batch
input_x = test_dataset[idx][0].unsqueeze(0)
#feature_maps是有两个元素的列表，分别表示第一层和第二层卷积的所有特征图
feature_maps = net.retrieve_features(input_x)
plt.figure(figsize = (10, 7))
#打印出4个特征图
for i in range(4):
    plt.subplot(1,4,i + 1)
    plt.imshow(feature_maps[0][0, i,...].data.numpy())

#绘制第二层的卷积核，每一列对应一个卷积核，一共有8个卷积核
plt.figure(figsize = (15, 10))
for i in range(4):
    for j in range(8):
        plt.subplot(4, 8, i * 8 + j + 1)
        plt.imshow(net.conv2.weight.data.numpy()[j, i,...])

#绘制第二层的特征图，一共有8个
plt.figure(figsize = (10, 7))
for i in range(8):
    plt.subplot(2,4,i + 1)
    plt.imshow(feature_maps[1][0, i,...].data.numpy())


"""5.3.3 卷积神经网络的健壮性实验"""
#提取test_dataset中第idx个批次第0个图第0个通道对应的图像，定义为a
a = test_dataset[idx][0][0]
#将平移后的新图像放到b中，根据a给b赋值
b = torch.zeros(a.size()) #全0的28×28的矩阵
w = 3 #平移的长度为3个像素
#对于b中的任意像素i,j，等于a中的i,j+w位置的像素
for i in range(a.size()[0]):
    for j in range(0, a.size()[1] - w):
        b[i, j] = a[i, j + w]
#将b画出来
muteimg = b.numpy()
plt.imshow(muteimg)

#把b输入神经网络，得到分类结果pred（prediction是预测的每一个类别的概率的对数值）并打印
prediction = net(b.unsqueeze(0).unsqueeze(0))
pred = torch.max(prediction.data, 1)[1]
print(pred)
#提取b对应的featuremap结果
feature_maps = net.retrieve_features(b.unsqueeze(0).unsqueeze(0))
plt.figure(figsize = (10, 7))
for i in range(4):
    plt.subplot(1,4,i + 1)
    plt.imshow(feature_maps[0][0, i,...].data.numpy())
    plt.figure(figsize = (10, 7))
for i in range(8):
    plt.subplot(2,4,i + 1)
    plt.imshow(feature_maps[1][0, i,...].data.numpy())