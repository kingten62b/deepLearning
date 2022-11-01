import numpy as np
import torch

# 类型转换
x_tensor = torch.randn(2,3)
y_numpy = np.random.randn(2,3)
x_numpy = x_tensor.numpy()
y_tensor = torch.from_numpy(y_numpy)
print(x_numpy)
print(y_tensor)
