import os
'''
配置文件
'''

'''num_worker设为0,意味着每一轮迭代时,dataloader不再有自主加载数据到RAM这一步骤,
而是在RAM中找batch,找不到时再加载相应的batch'''
NUM_WORKERS=0   # dataloader线程数

TRAIN_SCALE=0.8 # 训练集所占比例
BATCH_SIZE=10   # 每次迭代使用多少个样本
EPOCHS=20       # 训练的总循环周期
IMG_SIZE=64     # 输入图像尺寸
# IMG_SIZE=224     # 输入图像尺寸 (train_test使用)
DEFAULT_MODEL="face_modle.pkl" # 模型名称

# 工作区文件夹路径
PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

# 数据集
DATA_TRAIN = os.path.join(PROJECT_PATH, "data/img")

# 数据集  CALFW  图片总数12174 人数4025
# DATA_TRAIN = os.path.join(PROJECT_PATH, "F:\\data\\CALFW\\img_classify")

# 模型保存地址
DATA_MODEL = os.path.join(PROJECT_PATH, "data/model")
