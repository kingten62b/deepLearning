#PyTorch用的包
import torch
import torch.nn as nn
import torch.optim
#自然语言处理相关的包
import re #正则表达式的包
import jieba #结巴分词包
from collections import Counter #搜集器，可以让统计词频更简单
#绘图、计算用的包
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

#------------------ 数据处理----------------------
#数据来源文件
good_file = 'D:/myproject/py/deepLearning/04/data/good.txt'
bad_file = 'D:/myproject/py/deepLearning/04/data/bad.txt'
# good_file = '04/data/good.txt'
# bad_file = '04/data/bad.txt'
#将文本中的标点符号过滤掉
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return(sentence)
#扫描所有的文本，分词并建立词典，分出正向还是负向的评论，is_filter可以过滤是否筛选掉标点符号
def Prepare_data(good_file, bad_file, is_filter = True):
    all_words = [] #存储所有的单词
    pos_sentences = [] #存储正向的评论
    neg_sentences = [] #存储负向的评论
    with open(good_file, 'r', encoding='utf-8') as fr: #打开文件，用于文件的读写操作，省去了关闭文件的麻烦。
        for idx, line in enumerate(fr):
            if is_filter:
                #过滤标点符号
                line = filter_punc(line)
            #分词
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)
    print('{0} 包含 {1} 行，{2} 个词.'.format(good_file, idx+1, len(all_words)))
    count = len(all_words)
    with open(bad_file, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
                words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print('{0} 包含 {1} 行，{2} 个词.'.format(bad_file, idx+1, len(all_words)-count))
    #建立词典，diction的每一项为{w:[id, 单词出现次数]}
    diction = {}
    cnt = Counter(all_words) #计算出字符串或者列表等中不同元素出现的个数，返回值可以理解为一个字典
    for word, freq in cnt.items():
        diction[word] = [len(diction), freq]
    print('字典大小：{}'.format(len(diction)))
    return(pos_sentences, neg_sentences, diction)
#调用Prepare_data，完成数据处理工作
pos_sentences, neg_sentences, diction = Prepare_data(good_file, bad_file, True)
st = sorted([(v[1], w) for w, v in diction.items()])

#根据单词返还单词的编码
def word2index(word, diction):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return(value)
#根据编码获得对应的单词
def index2word(index, diction):
    for w,v in diction.items():
        if v[0] == index:
            return(w)
    return(None)

#---------------文本数据向量化---------------
#输入一个句子和相应的词典，得到这个句子的向量化表示
#向量的尺寸为词典中词汇的个数，i位置上面的数值为第i个单词在sentence中出现的频率
def sentence2vec(sentence, dictionary):
    vector = np.zeros(len(dictionary))
    for l in sentence:
        vector[l] += 1
    return(1.0 * vector / len(sentence))
#遍历所有句子，将每一个词映射成编码
dataset = [] #数据集
labels = [] #标签
sentences = [] #原始句子，调试用
#处理正向评论
for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(0) #正标签为0
    sentences.append(sentence)
#处理负向评论
for sentence in neg_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(1) #负标签为1
    sentences.append(sentence)
#打乱所有的数据顺序，形成数据集
#indices为所有数据下标的全排列
indices = np.random.permutation(len(dataset))
#根据打乱的下标，重新生成数据集dataset、标签集labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]

#---------------划分测试集、训练集----------------
#将整个数据集划分为训练集、校验集和测试集，其中校验集和测试集的长度都是整个数据集的十分之一
test_size = int(len(dataset)//10)
train_data = dataset[2 * test_size :]
train_label = labels[2 * test_size :]
valid_data = dataset[: test_size]
valid_label = labels[: test_size]
test_data = dataset[test_size : 2 * test_size]
test_label = labels[test_size : 2 * test_size]

# --------------建立神经网络--------------------
#一个简单的前馈神经网络，共3层
#第一层为线性层，加一个非线性ReLU，第二层为线性层，中间有10个隐含层神经元
#输入维度为词典的大小：每一段评论的词袋模型
model = nn.Sequential(
    nn.Linear(len(diction), 10),
    nn.ReLU(),
    nn.Linear(10, 2),
    nn.LogSoftmax(dim=1),
)
#自定义的计算一组数据分类准确度的函数
#predictions为模型给出的预测结果，labels为数据中的标签。比较二者以确定整个神经网络当前的表现
def rightness(predictions, labels):
    '''计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据中的正确答案'''
    #对于任意一行（一个样本）的输出值的第1个维度求最大，得到每一行最大元素的下标
    pred = torch.max(predictions.data, 1)[1]
    #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素

#损失函数为交叉熵
cost = torch.nn.NLLLoss()
#优化算法为SGD，可以自动调节学习率
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
records = []
#循环10个epoch
losses = []
for epoch in range(10):
    for i, data in enumerate(zip(train_data, train_label)):
        x, y = data
        #将输入的数据进行适当的变形，主要是要多出一个batch_size的维度，即第一个为1的维度
        x = torch.tensor(x,requires_grad = True, dtype = torch.float).view(1,-1) #x的尺寸：batch_size=1, len_dictionary
        #标签也要加一层外衣以变成1*1的张量
        y = torch.tensor(np.array([y]), dtype = torch.long)
        #y的尺寸：batch_size=1, 1
        #清空梯度
        optimizer.zero_grad()
        #模型预测
        predict = model(x)
        #计算损失函数
        loss = cost(predict, y)
        #将损失函数数值加入列表中
        losses.append(loss.data.numpy())
        #开始进行梯度反传
        loss.backward()
        #开始对参数进行一步优化
        optimizer.step()
        #每隔3000步，跑一下校验集的数据，输出临时结果
        if i % 3000 == 0:
            val_losses = []
            rights = []
            #在所有校验集上实验
            for j, val in enumerate(zip(valid_data, valid_label)):
                x, y = val
                x = torch.tensor(x, requires_grad = True, dtype = torch.float).view(1,-1)
                y = torch.tensor(np.array([y]), dtype = torch.long)
                predict = model(x)
                #调用rightness函数计算准确度
                right = rightness(predict, y)
                rights.append(right)
                loss = cost(predict, y)
                val_losses.append(loss.data.numpy())
            #将校验集上的平均准确度计算出来
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('第{}轮，训练损失：{:.2f}，校验损失：{:.2f}，校验准确率：{:.2f}'.format(epoch,
                np.mean(losses), np.mean(val_losses), right_ratio))
            records.append([np.mean(losses), np.mean(val_losses), right_ratio])


y = np.array(records)
y_TrainLoss = y[:,0]
y_val_losses = y[:,1]
y_right_ratio = y[:,2]
x = np.arange(len(y_TrainLoss))
plt.figure(figsize = (10, 7)) #设定绘图窗口大小
line0, = plt.plot(x, y_TrainLoss) #绘制
line1, = plt.plot(x, y_val_losses) #绘制
line2, = plt.plot(x, y_right_ratio) #绘制
plt.xlabel('septs') #更改坐标轴标注
plt.ylabel('loss&accuracy') #更改坐标轴标注
plt.legend([line0, line1, line2],['Train Loss','Valid Losses','valid Accuracy']) #绘制图例
plt.show()



#在测试集上分批运行，并计算总的正确率
vals = [] #记录准确率所用列表
#对测试数据集进行循环
for data, target in zip(test_data, test_label):
    data, target =torch.tensor(data, dtype = torch.float).view(1,-1), torch.tensor(np.array([target]), dtype = torch.long)
    output = model(data) #将特征数据输入网络，得到分类的输出
    val = rightness(output, target) #获得正确样本数以及总样本数
    vals.append(val) #记录结果
#计算准确率
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0 * rights[0].data.numpy() / rights[1]
print("准确率",right_rate)

