import os
# 工作区文件夹路径
PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

# 训练数据集
DATA_TRAIN = os.path.join(PROJECT_PATH, "data/train")
# 验证数据集
DATA_TEST = os.path.join(PROJECT_PATH, "data/test")
# 模型保存地址
DATA_MODEL = os.path.join(PROJECT_PATH, "data/model")