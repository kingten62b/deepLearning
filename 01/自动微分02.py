'''自动微分'''
import torch

s = torch.tensor([[0.01, 0.02]], requires_grad = True)
x = torch.ones(2, 2, requires_grad = True)
for i in range(10):
    s = s.mm(x)
z = torch.mean(s)

z.backward()
print(x.grad)
print(s.grad)