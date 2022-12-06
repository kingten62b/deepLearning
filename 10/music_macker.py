#导入必需的依赖包
#与PyTorch相关的包
import torch
import torch.utils.data as DataSet
import torch.nn as nn
import torch.optim as optim
#导入MIDI音乐处理的包
from mido import MidiFile, MidiTrack, Message
#导入计算与绘图必需的包
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#从硬盘中读入MIDI音乐文件
# mid = MidiFile('./music/krebs.mid') #a Mozart piece
mid = MidiFile('F:\\data\\mid_music\\krebs.mid') #a Mozart piece
notes = []
time = float(0)
prev = float(0)
original = [] #original记载了原始的message数据，以便后面进行比较
#对MIDI文件中所有的消息进行循环
for msg in mid:
    #时间的单位是秒，而不是帧
    time += msg.time
    #如果当前消息不是描述信息
    if not msg.is_meta:
        #仅提炼第一个channel的音符
        if msg.channel == 0:
        #如果当前音符为打开的
            if msg.type == 'note_on':
                #获得消息中的信息（编码在字节中）
                note = msg.bytes()
                #我们仅对音符信息感兴趣。音符信息按如下形式记录 [type, note, velocity]
                note = note[1:3] #操作完这一步后，note[0]存音符，note[1]存速度（力度）
                #note[2]存距上一个message的时间间隔
                note.append(time - prev)
                prev = time
                #将音符添加到列表notes中
                notes.append(note)
                #在原始列表中保留这些音符
                original.append([i for i in note])

#note和velocity都可以看作类型变量
#time为float类型，按照区间将其转化成离散的类型变量
#首先，找到time变量的取值区间并进行划分。由于大量message的time为0，因此把0归为一个特别的类
intervals = 10
values = np.array([i[2] for i in notes])
max_t = np.amax(values) #区间中的最大值
min_t = np.amin(values[values > 0]) #区间中的最小值
interval = 1.0 * (max_t - min_t) / intervals
#接下来，将每一个message编码成3个独热向量，将这3个向量合并到一起就构成了slot向量
dataset = []
for note in notes:
    slot = np.zeros(89 + 128 + 12)
    #由于note介于24~112之间，因此减24
    ind1 = note[0]-24
    ind2 = note[1]
    #由于message中有大量time=0的情况，因此将0归为单独的一类，其他的一律按照区间划分
    ind3 = int((note[2] - min_t) / interval + 1) if note[2] > 0 else 0
    slot[ind1] = 1
    slot[89 + ind2] = 1
    slot[89 + 128 + ind3] = 1
    #将处理后得到的slot数组加入dataset中
    dataset.append(slot)

#生成训练集和校验集
X = []
Y = []
#首先，按照预测的模式，将原始数据生成一对一对的训练数据
n_prev = 30 #滑动窗口长度为30
#对数据中的所有数据进行循环
for i in range(len(dataset)-n_prev):
    #往后取n_prev个note作为输入属性
    x = dataset[i:i+n_prev]
    #将第n_prev+1个note（编码前）作为目标属性
    y = notes[i+n_prev]
    #注意time要转化成类别的形式
    ind3 = int((y[2] - min_t) / interval + 1) if y[2] > 0 else 0
    y[2] = ind3
    #将X和Y加入数据集中
    X.append(x)
    Y.append(y)
#将数据集中的前n_prev个音符作为种子，用于生成音乐
seed = dataset[0:n_prev]
#将所有数据顺序打乱重排
idx = np.random.permutation(range(len(X)))
#形成训练与校验数据集列表
X = [X[i] for i in idx]
Y = [Y[i] for i in idx]
#从中切分出1/10的数据放入校验集
validX = X[: len(X) // 10]
X = X[len(X) // 10 :]
validY = Y[: len(Y) // 10]
Y = Y[len(Y) // 10 :]
'''将列表再转化为dataset，并用dataloader来加载数据。dataloader是PyTorch开发采用的一套管理数据的方法。通常数据的存储放在dataset中，而对数据的调用则是通过dataloader完成的。同时，在进行预处理时，系统已经自动将数据打包成批（batch），每次调用都提取出一批（包含多条记录）。从dataloader中输出的每一个元素都是一个(x,y)元组，其中x为输入的张量，y为标签。x和y的第一个维度都是batch_size大小。'''
'''一批包含30个数据记录。这个数字越大，系统在训练的时候，每一个周期要处理的数据就越多，处理就越快，但总的数据量会减少。'''
batch_size = 30
#形成训练集
train_ds = DataSet.TensorDataset(torch.FloatTensor(np.array(X, dtype = float)), torch.LongTensor(np.array(Y)))
#形成数据加载器
train_loader = DataSet.DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers=0)
#校验数据
valid_ds = DataSet.TensorDataset(torch.FloatTensor(np.array(validX, dtype = float)), torch.LongTensor(np.array(validY)))
valid_loader = DataSet.DataLoader(valid_ds, batch_size = batch_size, shuffle = True, num_workers=0)

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers=1):
        super(LSTMNetwork, self).__init__()
        self.n_layers = n_layers        # 网络层数 = 1
        self.hidden_size = hidden_size  # 隐藏记忆单元个数
        self.out_size = out_size        # 输出维度
        #一层LSTM单元
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first = True)
        #一个Dropout部件，以0.2的概率dropout
        self.dropout = nn.Dropout(0.2)
        #一个全连接层
        self.fc = nn.Linear(hidden_size, out_size)  # 输出层
        #对数Softmax层
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden=None):
        #神经网络的每一步运算
        hhh1 = hidden[0] #读入隐含层的初始信息
        #完成一步LSTM运算
        #input的尺寸为：batch_size, time_step, input_size
        output, hhh1 = self.lstm(input, hhh1) #input:batchsize*timestep*3
        #对神经元输出的结果进行dropout
        output = self.dropout(output)
        #取出最后一个时刻的隐含层输出值
        #output的尺寸为：batch_size, time_step, hidden_size
        output = output[:, -1, ...]
        #此时，output的尺寸为：batch_size, hidden_size
        #输入一个全连接层
        out = self.fc(output)
        #out的尺寸为：batch_size, output_size
        #将out的最后一个维度分割成3份x, y, z，分别对应了对note，velocity以及time的预测
        # note [0]存音符（note） note [1]存速度（velocity） note [2]存距离上一个message的时间间隔 
        x = self.softmax(out[:, :89])
        y = self.softmax(out[:, 89: 89 + 128])
        z = self.softmax(out[:, 89 + 128:])
        #x的尺寸为batch_size, 89
        #y的尺寸为batch_size, 128
        #z的尺寸为batch_size, 11
        #返回x,y,z
        return (x,y,z)

    def initHidden(self, batch_size):
        #将隐含层单元变量全部初始化为0
        #注意尺寸是：layer_size, batch_size, hidden_size
        out = []
        hidden1= torch.zeros(1, batch_size, self.hidden_size)
        cell1= torch.zeros(1, batch_size, self.hidden_size)
        out.append((hidden1, cell1))
        return out

def criterion(outputs, target):
    #为本模型自定义的损失函数，由3部分组成，每部分都是一个交叉熵损失函数
    #分别对应note、velocity和time的交叉熵
    x, y, z = outputs
    loss_f = nn.NLLLoss()
    loss1 = loss_f(x, target[:, 0])
    loss2 = loss_f(y, target[:, 1])
    loss3 = loss_f(z, target[:, 2])
    return loss1 + loss2 + loss3

def rightness(predictions, labels):
    '''计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据中的正确答案'''
    #对于任意一行（一个样本）的输出值的第1个维度求最大，得到每一行最大元素的下标
    pred = torch.max(predictions.data, 1)[1]
    #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    rights = pred.eq(labels.data).sum()
    return rights, len(labels) #返回正确的数量和这次一共比较了多少元素

#定义一个LSTM，其中输入层和输出层的单元个数取决于每个变量的类型取值范围
lstm = LSTMNetwork(89 + 128 + 12, 128, 89 + 128 + 12)
optimizer = optim.Adam(lstm.parameters(), lr=0.001)
num_epochs = 100
train_losses = []
valid_losses = []
records = []
#开始训练循环
for epoch in range(num_epochs):
    train_loss = []
    #开始遍历加载器中的数据
    for batch, data in enumerate(train_loader):
        #batch为数字，表示已经进行了第几个batch
        #data为一个二元组，分别存储了一条数据记录的输入和标签
        #每个数据的第一个维度都是batch_size = 30的数组
        lstm.train() #标志LSTM当前处于训练阶段，Dropout开始起作用
        init_hidden = lstm.initHidden(len(data[0])) #初始化LSTM的隐含单元变量
        optimizer.zero_grad()
        x, y = data[0], data[1] #从数据中提炼出输入和输出对
        outputs = lstm(x, init_hidden) #输入LSTM，产生输出outputs
        loss = criterion(outputs, y) #代入损失函数并产生loss
        train_loss.append(loss.data.numpy()) #记录loss
        loss.backward() #反向传播
        optimizer.step() #梯度更新
    if 0 == 0:
        #在校验集上运行一遍，并计算在校验集上的分类准确率
        valid_loss = []
        lstm.eval() #将模型标志为测试状态，关闭dropout的作用
        rights = []
        #遍历加载器加载进来的每一个元素
        for batch, data in enumerate(valid_loader):
            init_hidden = lstm.initHidden(len(data[0]))
            #完成LSTM的计算
            # x, y = Variable(data[0]), Variable(data[1])
            x, y = data[0], data[1]
            #x的尺寸：batch_size, length_sequence, input_size
            #y的尺寸：batch_size, (data_dimension1=89+ data_dimension2=128+ data_dimension3=12)
            outputs = lstm(x, init_hidden)
            #outputs: (batch_size*89, batch_size*128, batch_size*11)
            loss = criterion(outputs, y)
            valid_loss.append(loss.data.numpy())
            #计算每个指标的分类准确度
            right1 = rightness(outputs[0], y[:, 0])
            right2 = rightness(outputs[1], y[:, 1])
            right3 = rightness(outputs[2], y[:, 2])
            rights.append((right1[0] + right2[0] + right3[0]) * 1.0 / (right1[1] + right2[1] + right3[1]))
        #打印结果
        print('第{}轮，训练Loss：{:.2f}，校验Loss：{:.2f}，校验准确度：{:.2f}'.format(epoch,
                np.mean(train_loss),
                np.mean(valid_loss),
                np.mean(rights)
                ))
        records.append([np.mean(train_loss), np.mean(valid_loss), np.mean(rights)])

#绘制训练过程中的Loss曲线
a = [i[0] for i in records]
b = [i[1] for i in records]
c = [i[2] * 10 for i in records]
plt.plot(a, '-', label = 'Train Loss')
plt.plot(b, '-', label = 'Validation Loss')
plt.plot(c, '-', label = '10 * Accuracy')
plt.legend()
plt.show()