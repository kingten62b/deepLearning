# -*- coding: utf-8 -*-
import cv2
import os

def catch_video(tag, window_name='catch face', camera_idx=0):
    cv2.namedWindow(window_name)
    # 摄像头
    cap = cv2.VideoCapture(camera_idx)
    while cap.isOpened():
        # 逐帧捕获
        ok, frame = cap.read()
        if not ok:
            break
        # 抓取人脸
        catch_face(frame, tag)
        # 输入'q'退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
  # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

def catch_face(frame, tag):
    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("openCV/faceDemo/cv2data/haarcascade_frontalface_alt2.xml")
    # 人脸边框的颜色
    color = (0, 255, 0)
    # 将当前帧转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    num = 1
    if len(face_rects) > 0: # 大于0则检测到人脸
        # 图片帧中有多个图片，框出每一个人脸
        for face_rects in face_rects:
            x, y, w, h = face_rects
            # image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            # 保存人脸图像
            # save_face(image, tag, num)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            num += 1

# def save_face(image, tag, num):
#   # DATA_TRAIN为抓取的人脸存放目录，如果目录不存在则创建
#     makedir_exist_ok(os.path.join(DATA_TRAIN, str(tag)))
#     img_name = os.path.join(DATA_TRAIN, str(tag), '{}_{}.jpg'.format(int(time.time()), num))
#     # 保存人脸图像到指定的位置, 其中会创建一个tag对应的目录，用于后面的分类训练
#     cv2.imwrite(img_name, image)

# PROJECT_PATH = os.path.abspath(
#     os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

# # 训练数据集
# DATA_TRAIN = os.path.join(PROJECT_PATH, "data/train")
# # 验证数据集
# DATA_TEST = os.path.join(PROJECT_PATH, "data/test")
# # 模型保存地址
# DATA_MODEL = os.path.join(PROJECT_PATH, "data/model")

if __name__ == '__main__':
    catch_video("liu")