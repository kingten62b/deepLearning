import os
'''
配置文件
'''

NUM_WORKERS=1   # 几个线程
TRAIN_SCALE=0.8 # 训练集所占比例
BATCH_SIZE=20   # 每次迭代使用多少个样本
EPOCHS=20       # 训练的总循环周期
IMG_SIZE=64     # 输入图像尺寸
DEFAULT_MODEL="face_modle.pkl" # 模型名称

# 工作区文件夹路径
PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

# 训练集数据集
DATA_TRAIN = os.path.join(PROJECT_PATH, "data/train")
# 验证集数据
DATA_VALIDATION = os.path.join(PROJECT_PATH, "data/test")
# 模型保存地址
DATA_MODEL = os.path.join(PROJECT_PATH, "data/model")
