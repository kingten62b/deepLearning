import os
'''
配置文件
'''

NUM_WORKERS=1   # 几个线程
TRAIN_SCALE=0.8 # 训练集所占比例
BATCH_SIZE=20   # 每次迭代使用多少个样本
EPOCHS=25       # 训练的总循环周期
IMG_SIZE=64     # 输入图像尺寸
# IMG_SIZE=224     # 输入图像尺寸
DEFAULT_MODEL="face_modle.pkl" # 模型名称

# 工作区文件夹路径
PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

# 数据集
DATA_TRAIN = os.path.join(PROJECT_PATH, "data/img")

# 数据集  CALFW  图片总数12174 ren4024
# DATA_TRAIN = os.path.join(PROJECT_PATH, "F:\\data\\CALFW\\img_classify")

# 模型保存地址
DATA_MODEL = os.path.join(PROJECT_PATH, "data/model")
