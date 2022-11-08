import numpy as np
import pandas as pd #读取csv文件的库
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
#让输出图形直接在Notebook中显示
# %matplotlib inline

data_path = 'D:/myproject/py/deepLearning/02/hour.csv' #读取数据到内存，rides为一个dataframe对象
rides = pd.read_csv(data_path)


# -------------------类型变量的独热编码--------------------------------------
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday'] #所有类型编码变量的名称
for each in dummy_fields:
    #取出所有类型变量，并将它们转变为独热编码
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    #将新的独热编码变量与原有的所有变量合并到一起
    rides = pd.concat([rides, dummies], axis=1)
#将原来的类型变量从数据表中删除
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday','hr'] #要删除的类型变量的名称
data = rides.drop(fields_to_drop, axis=1) #将它们从数据库的变量中删除


# ---------------------数值类型变量的标准化------------------------------------
quant_features = ['cnt', 'temp', 'hum', 'windspeed'] #数值类型变量的名称
scaled_features = {} #将每一个变量的均值和方差都存储到scaled_features变量中
for each in quant_features:
    #计算这些变量的均值和方差
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    #对每一个变量进行标准化
    data.loc[:, each] = (data[each] - mean)/std


test_data = data[-21*24:] #选出训练集
train_data = data[:-21*24] #选出测试集
#目标列包含的字段
target_fields = ['cnt','casual', 'registered']
#训练集划分成特征变量列和目标特征列
features, targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
#测试集划分成特征变量列和目标特征列
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
#将数据类型转换为NumPy数组
X = features.values #将数据从pandas dataframe转换为NumPy
Y = targets['cnt'].values
Y = Y.astype(float)
Y = np.reshape(Y, [len(Y),1])
losses = []


#定义神经网络架构，features.shape[1]个输入层单元，10个隐含层，1个输出层
input_size = features.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128
neu = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)
cost = torch.nn.MSELoss()   #计算均方误差的损失函数
optimizer = torch.optim.SGD(neu.parameters(), lr = 0.01)    #torch.optim.SGD()调用了PyTorch自带的随机梯度下降算法   学习率lr=0.01


#神经网络训练循环
losses = []
for i in range(1000):
    #每128个样本点被划分为一批，在循环的时候一批一批地读取
    batch_loss = []
    #start和end分别是提取一批数据的起始和终止下标
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = torch.FloatTensor(X[start:end])
        yy = torch.FloatTensor(Y[start:end])
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    #每隔100步输出损失值
    if i % 100==0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))
#打印输出损失值
plt.plot(np.arange(len(losses))*100,losses)
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()


targets = test_targets['cnt'] #读取测试集的cnt数值
targets = targets.values.reshape([len(targets),1]) #将数据转换成合适的tensor形式
targets = targets.astype(float) #保证数据为实数
x = torch.FloatTensor(test_features.values)
y = torch.FloatTensor(targets)
#用神经网络进行预测
predict = neu(x)
predict = predict.data.numpy()
fig, ax = plt.subplots(figsize = (10, 7))
mean, std = scaled_features['cnt']
ax.plot(predict * std + mean, label='Prediction')
ax.plot(targets * std + mean, label='Data')
ax.legend()
ax.set_xlabel('Date-time')
ax.set_ylabel('Counts')
dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()



def feature(X, net):
    #定义一个函数，用于提取网络的权重信息，所有的网络参数信息全部存储在neu的named_parameters集合中
    X = torch.from_numpy(X).type(torch.FloatTensor)
    dic = dict(net.named_parameters()) #提取这个集合
    weights = dic['0.weight'] #可以按照“层数.名称”来索引集合中的相应参数值
    biases = dic['0.bias']
    h = torch.sigmoid(X.mm(weights.t()) + biases.expand([len(X), len(biases)])) #隐含层的计算过程
    return h #输出层的计算


bool1 = rides['dteday'] == '2012-12-22'
bool2 = rides['dteday'] == '2012-12-23'
bool3 = rides['dteday'] == '2012-12-24'
#将3个布尔型数组求与
bools = [any(tup) for tup in zip(bool1,bool2,bool3) ]
#将相应的变量取出来
subset = test_features.loc[rides[bools].index]
subtargets = test_targets.loc[rides[bools].index]
subtargets = subtargets['cnt']
subtargets = subtargets.values.reshape([len(subtargets),1])


#将数据输入到神经网络中，读取隐含层神经元的激活数值，存入results中
results = feature(subset.values, neu).data.numpy()
#这些数据对应的预测值（输出层）
predict = neu(torch.FloatTensor(subset.values)).data.numpy()
#将预测值还原为原始数据的数值范围
mean, std = scaled_features['cnt']
predict = predict * std + mean
subtargets = subtargets * std + mean


#将所有的神经元激活水平画在同一张图上
fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(results[:,:],'.:',alpha = 0.1)
ax.plot((predict - min(predict)) / (max(predict) - min(predict)),'bo-',label='Prediction')
ax.plot((subtargets - min(predict)) / (max(predict) - min(predict)),'ro-',label='Real')
ax.plot(results[:, 5],'.:',alpha=1,label='Neuro 6')
ax.set_xlim(right=len(predict))
ax.legend()
plt.ylabel('Normalized Values')
dates = pd.to_datetime(rides.loc[subset.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
