from cgi import print_arguments
import torch

# 创建一个尺寸为(5, 3)的二阶张量（也就是5行3列的矩阵）,每个元素是[0, 1]区间中的一个随机数
x=torch.rand(5,3)
# 创建一个尺寸为(5, 3)、内容全是1的张量
y=torch.ones(5,3)
# 创建一个尺寸为(2, 5, 3)、内容全是0的张量
z=torch.zeros(2,5,3)
print(x)
# print(y)
# print(z)
# print(z[1])
# print(z[:,0,0])
# print(x+y)
# print((x+y).t())    #矩阵转置.t()
print(x.mm(y.t()))    #矩阵相乘