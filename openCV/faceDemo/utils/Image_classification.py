import os,shutil

'''将CALFW数据集分类整理成子文件夹'''
INPUT_PATH = 'F:\\data\\CALFW\\images' # 分类整理图片文件夹路径
OUTPUT_PATH = 'F:\\data\\CALFW\\img_classify' # 分类整理后输出

ls = os.listdir(INPUT_PATH)
print ("文件总数：",len(ls))

# 提取人名并去重
# Aaron_Eckhart_0001.jpg --> Aaron_Eckhart
name = set()
for each in ls:
    each = each[:(each.rindex('_'))] # 将文件名按人名切分
    name.add(each)
print("人脸总数：",len(name))
# 创建按人名分类的文件夹
for each in name:
    dir_path = os.path.join(OUTPUT_PATH, each)
    os.makedirs(dir_path)
# 将图片复制到对应文件夹
for each in ls:
    img_from_path = os.path.join(INPUT_PATH, each)
    img_to_path = os.path.join(OUTPUT_PATH, each[:(each.rindex('_'))], each)
    print(img_from_path,'\n' ,img_to_path)
    shutil.copy(img_from_path, img_to_path)
