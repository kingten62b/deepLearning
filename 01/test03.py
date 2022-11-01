import torch

x=torch.rand(3,3,3)
y=torch.ones(3,3,3)
# 张量放置到GPU上运算
if torch.cuda.is_available():
    x=x.cuda()
    y=y.cuda()
    print(x+y)
# 张量放置到CPU上运算
x=x.cpu()
y=y.cpu()
print(x+y)