# -*- encoding: utf8 -*-
import config
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_transform():
    return transforms.Compose([
            # 图像缩放到32 x 32
            transforms.Resize(32),
            # 中心裁剪 32 x 32
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            # 对每个像素点进行归一化
            transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                 std=[0.2, 0.2, 0.2])
        ])

def get_dataset(batch_size=10, num_workers=1):
    data_transform = get_transform()
    # load训练集图片
    train_dataset = ImageFolder(root=config.DATA_TRAIN, transform=data_transform)
    # load验证集图片
    test_dataset = ImageFolder(root=config.DATA_TEST, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader