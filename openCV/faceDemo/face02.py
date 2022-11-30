import cv2
import torch
import config
import numpy as np
from model import Net
import os
from data_pretreatment import get_dataset, get_transform
from PIL import Image, ImageDraw

FACE_LABEL = {
    0: "liu",
    1: "wu_jing",
    2: "wu_zhi_hao"
}

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE="cpu"
# print(DEVICE)

def recognize_video(window_name='face recognize', camera_idx=0):
    cv2.namedWindow(window_name)
    print("摄像头编号:{}".format(camera_idx))
    cap = cv2.VideoCapture(camera_idx)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        catch_frame = catch_face(frame)
        cv2.imshow(window_name, catch_frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def catch_face(frame):
    classfier = cv2.CascadeClassifier("openCV/faceDemo/cv2data/haarcascade_frontalface_alt2.xml")
    color = (0, 255, 0)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(face_rects) > 0:
        for face_rects in face_rects:
            x, y, w, h = face_rects
            image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            # opencv 2 PIL格式图片
            PIL_image = cv2pil(image)
            # 使用模型进行人脸识别
            label = predict_model(PIL_image)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            # 将人脸对应人名写到图片上, 中文名要加载中文字体库
            frame = paint_opencv(frame, FACE_LABEL[label], (x-10, y+h+10), (255,0,0))

    return frame

def cv2pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def predict_model(image):
    data_transform = get_transform()
    # 对图片进行预处理，同训练的时候一样
    image = data_transform(image)
    image = image.view(-1, 3, 32, 32)
    net = Net().to(DEVICE)
    # 加载模型参数权重
    net.load_state_dict(torch.load(os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL)))
    output = net(image.to(DEVICE))
    # 获取最大概率的下标
    pred = output.max(1, keepdim=True)[1]
    return pred.item()

# def paint_opencv(im, chinese, pos, color):
#     img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
#     # 引用字体库
#     font = ImageFont.truetype('/Library/Fonts/Songti.ttc', 20)
#     fillColor = color
#     position = pos
#     if not isinstance(chinese, unicode):
#         chinese = chinese.decode('utf-8')
#     draw = ImageDraw.Draw(img_PIL)
#     # 写上人脸对应的人名
#     draw.text(position, chinese, font=font, fill=fillColor)
#     img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
#     return img

def paint_opencv(im, chinese, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    fillColor = color
    position = pos
    draw = ImageDraw.Draw(img_PIL)
    # 写上人脸对应的人名
    draw.text(position, chinese, fill=fillColor)
    print(chinese)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

if __name__ == '__main__':
    recognize_video()
    # recognize_video(camera_idx="F:/t3856/Videos/test.mp4")