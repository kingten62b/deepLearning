'''自动微分'''
import torch

# requires_grad=True是为了保证它可以在反向传播算法的过程中获得梯度信息
x = torch.ones(2, 2, requires_grad=True)    
print(x)
y=x+2
# print(y.grad_fn)

# *自动微分变量或张量之间的按元素乘法（两个张量在对应位置上进行数值相乘，这与矩阵运算mm是完全不一样的）
z=y*y  
# print(z.grad_fn)

# mean()对矩阵的每个元素求和再除以元素的个数
t = torch.mean(z)
# print(t.grad_fn)

t.backward()
print(z.grad)
print(y.grad)
print(x.grad)