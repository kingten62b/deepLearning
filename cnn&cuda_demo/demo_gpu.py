import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from matplotlib import pyplot as plt
import time

DATA_PATH = "cnn&cuda_demo/mnist"   # 训练集路径

# 超参数
BATCH_SIZE = 50
LEARNING_RATE = 0.001
EPOCH_NUM = 3

# 数据准备
train_data = torchvision.datasets.MNIST(
    root=DATA_PATH,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_data = torchvision.datasets.MNIST(root=DATA_PATH, train=False)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 修改1：将数据改成CUDA可识别的格式
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255.
test_y = test_data.test_labels[:2000].cuda()


# 网络构建
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        conv1_res = self.conv1(x)
        conv2_res = self.conv2(conv1_res)
        out = conv2_res.view(conv2_res.size(0), -1)
        output = self.out(out)
        return output


# 网络新建与训练
cnn = CNN()
# 修改2：给网络设置cuda
cnn.cuda()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()

step_list = []
loss_list = []
accuracy_list = []
counter = 0

t1 = time.time()
for epoch in range(EPOCH_NUM):
    for step, (b_x, b_y) in enumerate(train_loader):
        # 修改3：将数据转换成CUDA可识别的格式
        predict_y = cnn(b_x.cuda())
        loss = loss_func(predict_y, b_y.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        counter += 1

        # 可视化代码
        if step % 25 == 0:
            test_output = cnn(test_x)
            # 修改4：将数据转换成CUDA可识别的格式
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = float(torch.sum(pred_y == test_y)) / float(test_y.size(0))
            # 修改5：将数据转换成Python可识别的格式
            print('epoch:', epoch, '|step:%4d' % step, '|loss:%6f' % loss.data.cpu(), '|accuracy:%4f' % accuracy)

            step_list.append(counter)
            loss_list.append(loss.data.cpu())
            accuracy_list.append(accuracy)

            plt.cla()
            plt.plot(step_list, loss_list, c='red', label='loss')
            plt.plot(step_list, accuracy_list, c='blue', label='accuracy')
            plt.legend(loc='best')
            plt.pause(0.1)
t2 = time.time()
print(t2 - t1)