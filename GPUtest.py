import torch
print(torch.__version__, torch.version.cuda)
print('gpu',torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_capability())