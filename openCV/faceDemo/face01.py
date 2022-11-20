# -*- coding: utf-8 -*-
import cv2

def catch_video(tag, window_name='catch face', camera_idx=0):
    cv2.namedWindow(window_name)
  # 视频来源，可以来自一段已存好的视频，也可以直接来自摄像头
    cap = cv2.VideoCapture(camera_idx)
    while cap.isOpened():
       # 读取一帧数据
        ok, frame = cap.read()
        if not ok:
            break
      # 抓取人脸的方法, 后面介绍
        # catch_face(frame, tag)
        # 输入'q'退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
  # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()