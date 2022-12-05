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
            # 图像缩放到128 x 128
            transforms.Resize(config.IMG_SIZE),
            # 中心裁剪 128 x 128
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            # 对每个像素点进行归一化
            # transforms.Normalize(mean=[0.4, 0.4, 0.4], # 均值
            #                      std=[0.2, 0.2, 0.2])  # 方差
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

def get_dataset(batch_size=10, num_workers=config.NUM_WORKERS):
    data_transform = get_transform()
    # # 图片
    # dataset = ImageFolder(root=config.DATA_TRAIN, 
    #                             transform=data_transform)
    # 训练集图片
    train_dataset = ImageFolder(root=config.DATA_TRAIN, 
                                transform=data_transform)
    # # 验证集图片
    # test_dataset = ImageFolder(root=config.DATA_TEST, 
    #                             transform=data_transform)
    # 训练数据集的加载器，自动将数据切分成批，顺序随机打乱
    # train_loader = DataLoader(dataset=train_dataset,
    #                             batch_size=batch_size,
    #                             shuffle=True,
    #                             num_workers=num_workers)
    # test_loader = DataLoader(dataset=test_dataset, 
    #                         batch_size=batch_size, 
    #                         shuffle=True, 
    #                         num_workers=num_workers)

    train_size = int(config.TRAIN_SCALE * len(train_dataset))
    validation_size = (len(train_dataset) - train_size) // 2
    test_size = len(train_dataset) - validation_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_dataset=train_dataset.dataset#这行很重要
    for images, labels in train_dataset.imgs:
        print(images)
        print(labels)

    print(len(train_dataset))
    print(train_dataset.class_to_idx) #查看子文件夹与标签的映射
    # return train_loader, test_loader

if __name__=="__main__":
    # train_loader, test_loader = get_dataset()
    get_dataset()