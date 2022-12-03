import os
import config
import torch
from data_pretreatment import get_dataset
from model import Net,rightness
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE="cpu"
# print(DEVICE)

def train_model():
    record = [] #记录训练数据集准确率/测试集准确率的容器,用于后续绘图
    train_loader, test_loader = get_dataset(batch_size=config.BATCH_SIZE)
    net = Net().to(DEVICE)
    # 使用Adam/SDG优化器
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(),
    #             lr=0.001,
    #             betas=(0.9, 0.999),
    #             eps=1e-08,
    #             weight_decay=0,
    #             amsgrad=False)
    for epoch in range(config.EPOCHS):
        train_rights = [] #记录训练数据集准确率
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x).to(DEVICE)
            # 使用最大似然 / log似然代价函数
            loss = F.nll_loss(output, y).to(DEVICE)          
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 清除显存
            # torch.cuda.empty_cache()

            if (step + 1) % 5 == 0:
                train_rights.append(rightness(output, y)) #将计算结果装到列表容器train_rights中
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                #开始在测试集上做循环，计算测试集上的准确度
                val_rights = [] #记录测试集准确率的容器
                for (x, y) in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    output = net(x)
                    val_rights.append(rightness(output, y)) #将计算结果装到列表容器val_rights中
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                print('训练周期: {} [{:.0f}%]\tLoss: {:.6f}\t训练集正确率: {:.3f}\t测试集正确率:{:.3f}'
                    .format(epoch+1, 100*(epoch+1)/config.EPOCHS, loss.item(), 
                            100*train_r[0]/train_r[1],
                            100*val_r[0]/val_r[1]))
                record.append(( (100-100*train_r[0]/train_r[1]).to("cpu"), (100-100*val_r[0]/val_r[1]).to("cpu") )) # 将数据移到CPU
    # 使用验证集查看模型效果
    test(net, test_loader)
    torch.save(net.state_dict(), os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL))
    return record

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            test_loss += F.nll_loss(output, y, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\ntest loss={:.4f}, accuracy={:.4f}\n'.format(test_loss, float(correct) / len(test_loader.dataset)))

if (__name__=="__main__"):
    record = train_model()
    #绘制训练过程的误差曲线，校验集和测试集上的错误率
    plt.figure(figsize = (10, 7))
    plt.plot(record) #record记载了每一个打印周期记录的训练集和校验集上的准确度
    plt.xlabel('Steps')
    plt.ylabel('Error rate')
    plt.show()