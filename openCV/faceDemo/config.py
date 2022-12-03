import os
'''
配置文件
'''

NUM_WORKERS=1 # 几个线程
BATCH_SIZE=10 # 每次迭代使用多少个样本
EPOCHS=40     # 训练的总循环周期
DEFAULT_MODEL="face_modle.pkl"

# 工作区文件夹路径
PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

# 训练数据集
DATA_TRAIN = os.path.join(PROJECT_PATH, "data/train")
# 测试数据 用于评估模型的准确率,不用于模型构建过程，否则会导致过渡拟合。
DATA_TEST = os.path.join(PROJECT_PATH, "data/test")
# 模型保存地址
DATA_MODEL = os.path.join(PROJECT_PATH, "data/model")
