import config
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

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
            transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                 std=[0.2, 0.2, 0.2])
        ])

def get_dataset(batch_size=10, num_workers=config.NUM_WORKERS):
    data_transform = get_transform()
    # 训练集图片
    train_dataset = ImageFolder(root=config.DATA_TRAIN, 
                                transform=data_transform)
    # 测试集图片
    test_dataset = ImageFolder(root=config.DATA_TEST, 
                                transform=data_transform)
    # 训练数据集的加载器，自动将数据切分成批，顺序随机打乱
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)
    print(train_dataset.class_to_idx) #查看子文件夹与标签的映射
    return train_loader, test_loader

if __name__=="__main__":
    train_loader, test_loader = get_dataset()