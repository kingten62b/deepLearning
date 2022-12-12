# deepLearning
深度学习记录

![image](b39322d5b31c87019f40cc25627f9e2f0508ff65.jpg)

opencv/FACEDEMO
===========================
一个简易人脸识别项目

###########环境依赖

pytorch1.13

opencv4.6

pillow

matplotlib

###########部署步骤
1. 安装环境依赖

2. 运行face01       //人脸检测,收集数据集,按Q退出

3. 运行train        //根据数据集训练模型

4. 运行face02       //人脸识别,按Q退出


###########目录结构描述

人脸识别项目  
openCV  
│  
├─data  
│  ├─img                    // 数据集存放目录  
│  │  
│  └─model  
│          face_modle.pkl   // 训练生成模型  
│  
├─faceDemo  
│  │  config.py             // 配置文件  
│  │  data_pretreatment.py  // 数据预处理       
│  │  face01.py             // 人脸检测,收集数据集  
│  │  face02.py             // 人脸识别  
│  │  model.py              // 定义卷积神经网络  
│  │  train.py              // 训练模型  
│  │  train_test.py         // resnet18模型迁移测试,训练(没啥用)  
│  │  
│  ├─cv2data  
│  │      haarcascade_frontalface_alt2.xml  // openCV人脸检测使用的分类器  
│  │      SimHei.ttf        // 中文字体(人脸对应人名写到图片上, 中文名要加载中文字体库)  
│  │  
│  ├─utils  
│  │      del_RGBA.py       // 过滤数据集中非RGB格式图片(部分数据集使用)  
│  │      Image_classification.py   // 数据集分类整理成子文件夹(部分数据集使用)  

###########鸣谢:  
alanzhu  https://github.com/sealzjh/face_recognize