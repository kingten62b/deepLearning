'''数据集含有RGBA(png)或者灰度图片需要去除'''
from PIL import Image
import os

INPUT_PATH = "F:\data\果蔬分类"     # 数据集路径
dirs = os.listdir(INPUT_PATH)
for each_dir in dirs:
    dir_path = os.path.join(INPUT_PATH, each_dir)
    imgs = os.listdir(dir_path)
    for each_img in imgs:
        img_path = os.path.join(INPUT_PATH, each_dir, each_img)
        print(img_path)
        try:
            Image.open(img_path).convert("RGB")
        except:
            