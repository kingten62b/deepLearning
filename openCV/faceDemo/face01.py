import cv2
import os
import time
import config

'''
人脸检测,收集数据集
'''

def catch_video(tag, window_name='catch face', camera_idx=0):
    """ catch_video 获取来自摄像头的视频流

    :param tag: 标签(人脸名)
    :param window_name: 窗口名称
    :param camera_idx: 摄像头编号;从文件播放视频时,用视频文件名替换摄像机索引
    :return None
    """
    cv2.namedWindow(window_name)
    # 设置摄像头
    print("摄像头编号:{}".format(camera_idx))
    cap = cv2.VideoCapture(camera_idx)
    # 检查摄像头是否启动
    if not cap.isOpened():
        print("无法启动摄像头")
        exit()
    while True:
        # 逐帧捕获
        ok, frame = cap.read()
        # 如果正确读取帧,ok为True
        if not ok:
            break
        # 人脸检测
        catch_face(frame, tag)
        # 显示结果帧e
        cv2.imshow(window_name, frame)
        # 输入'q'退出程序
        if cv2.waitKey(10) == ord('q'):
            break
  # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


def catch_face(frame, tag):
    """ catch_face 人脸检测,添加边框,并保存人脸图片

    :param frame: 需要检测人脸的帧
    :param tag: 标签(人脸名)
    :return num: 检测到的人脸数
    """
    # 使用openCV人脸识别分类器
    classfier = cv2.CascadeClassifier("openCV/faceDemo/cv2data/haarcascade_frontalface_alt2.xml")
    # 人脸边框的颜色
    color = (255, 0, 0)
    # 将当前帧转换成灰度图像,方便识别
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    # 人脸检测
    # scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
    # minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个);
    # minSize和maxSize用来限制得到的目标区域的范围
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    num = 1
    if len(face_rects) > 0: # 大于0则检测到人脸
        # 图片帧中有多个图片，框出每一个人脸
        for face_rects in face_rects:
            x, y, w, h = face_rects
            image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            # 保存人脸图像
            save_face(image, tag, num)
            # 绘制人脸边框
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            num += 1

def save_face(image, tag, num):
    # DATA_TRAIN为抓取的人脸存放目录，如果目录不存在则创建
    os.makedirs(os.path.join(config.DATA_TRAIN, str(tag)), exist_ok=True)
    # 图片文件名
    img_name = os.path.join(config.DATA_TRAIN, str(tag), '{}_{}.jpg'.format(int(time.time()), num))
    # 保存人脸图像到指定的位置
    cv2.imwrite(img_name, image)

if __name__ == '__main__':
    catch_video("luo_zhen_zhen")
    # catch_video(tag="liu_jia_tai",camera_idx="F:/t3856/Pictures/Camera Roll/liu01.mp4")