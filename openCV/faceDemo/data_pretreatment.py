import config
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch

'''
图片预处理
'''

def get_transform():
    return transforms.Compose([
            # 图像缩放到64 x 64
            transforms.Resize(config.IMG_SIZE),
            # 中心裁剪 64 x 64
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            # 对每个像素点进行归一化
            # transforms.Normalize(mean=[0.4, 0.4, 0.4], # 均值
            #                      std=[0.2, 0.2, 0.2])  # 方差
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))# 均值 方差
        ])

def get_dataset(batch_size=10, num_workers=config.NUM_WORKERS):
    data_transform = get_transform()
    # 先将所有图片放入训练集，之后按比例划分
    train_dataset = ImageFolder(root=config.DATA_TRAIN, 
                                transform=data_transform)
    # 按比例划分数据集
    sum_size = len(train_dataset)
    train_size = int(config.TRAIN_SCALE * sum_size)
    validation_size = (sum_size - train_size) // 2
    test_size = sum_size - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size,test_size])
    train_dataset, validation_dataset, test_dataset=train_dataset.dataset, validation_dataset.dataset, test_dataset.dataset

    # 数据集的加载器，顺序随机打乱
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    validation_loader = DataLoader(dataset=validation_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)

    print("sum_dataset:\t\t",sum_size, "\ntrain_dataset:\t\t",train_size, "\nvalidation_dataset:\t",validation_size, "\ntest_dataset:\t\t",test_size)
    print(train_dataset.class_to_idx) #查看子文件夹与标签的映射
    return train_loader, validation_loader, test_loader

if __name__=="__main__":
    get_dataset()