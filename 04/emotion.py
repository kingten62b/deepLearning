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
good_file = '04/data/good.txt'
bad_file = '04/data/bad.txt'
#将文本中的标点符号过滤掉
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return(sentence)
#扫描所有的文本，分词并建立词典，分出正向还是负向的评论，is_filter可以过滤是否筛选掉标点符号
def Prepare_data(good_file, bad_file, is_filter = True):
    all_words = [] #存储所有的单词
    pos_sentences = [] #存储正向的评论
    neg_sentences = [] #存储负向的评论
    with open(good_file, 'r', encoding='utf-8') as fr:
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
    cnt = Counter(all_words)
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
