import os
'''
配置文件
'''

'''num_worker设为0,意味着每一轮迭代时,dataloader不再有自主加载数据到RAM这一步骤,
而是在RAM中找batch,找不到时再加载相应的batch'''
NUM_WORKERS=0   # dataloader线程数

TRAIN_SCALE=0.9 # 训练集所占比例
BATCH_SIZE=20   # 每次迭代使用多少个样本
EPOCHS=20       # 训练的总循环周期
IMG_SIZE=64     # 输入图像尺寸
DEFAULT_MODEL="face_modle.pkl" # 模型名称

# 工作区文件夹路径
PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

# 数据集
DATA_TRAIN = os.path.join(PROJECT_PATH, "data/img")

# 数据集  CALFW  图片总数12174 人数4025
# DATA_TRAIN = os.path.join(PROJECT_PATH, "F:\\data\\CALFW\\img_classify")

# # 数据集  人脸识别 图片总数731 人数6
# DATA_TRAIN = os.path.join(PROJECT_PATH, "F:\\data\\人脸识别数据集\\images\\face")

# 数据集  果蔬分类 图片总数3355 种类36
# DATA_TRAIN = os.path.join(PROJECT_PATH, "F:\\data\\果蔬分类")

# 模型保存地址
DATA_MODEL = os.path.join(PROJECT_PATH, "data/model")
