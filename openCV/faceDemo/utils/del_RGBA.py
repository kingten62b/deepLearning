'''数据集含有不是RGB格式(RGBA或者灰度图片)需要去除'''
from PIL import Image
import os
import shutil

INPUT_PATH = "F:\data\果蔬分类"     # 数据集路径
OUTPUT_PATH = "F:\data\\trash"     # 数据集路径

count=0

dirs = os.listdir(INPUT_PATH)
for each_dir in dirs:
    dir_path = os.path.join(INPUT_PATH, each_dir)
    imgs = os.listdir(dir_path)
    for each_img in imgs:
        img_path = os.path.join(INPUT_PATH, each_dir, each_img)
        type = Image.open(img_path).mode
        if type != "RGB":
            print("move",img_path,type)
            count+=1
            shutil.move(img_path,os.path.join(OUTPUT_PATH,each_img))

print(count)
            