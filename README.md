# 🚗 OpenCV Lane Detection

使用 OpenCV 实现的车道线检测系统，适用于图像与视频的基本车道识别实验。项目涵盖图像预处理、边缘检测、ROI 提取、霍夫变换、线段筛选与拟合等步骤。

👉 项目地址：[lizhongzheng13/openCV_Lane_Detection](https://github.com/lizhongzheng13/openCV_Lane_Detection)

---

## 📁 项目结构
```text
openCV_Lane_Detection/
├── img.jpg           # 原始图像
├── video.mp4         # 原始视频
├── Canny_detect_edge.py # 边缘检测
├── roi_mask.py       # ROI 区域提取
├── hough_line.py     # 霍夫变换
├── filter_lines.py   # 离群值过滤
├── fit_line.py       # 最小二乘拟合
├── picture_detect_(draw_line).py     # 绘制拟合结果---最终图片检测代码
├── final_video_detect(mark_sequence).py  # 最终视频检测代码
├── *.jpg             # 中间结果图像
├── README.md
└── 从车道检测项目入门open cv说明文档.md     #必看！！！
```
## 📌 核心步骤

1.  图像灰度化 + 高斯模糊
2.  Canny 边缘检测
3.  提取感兴趣区域（ROI）
4.  使用霍夫变换检测线段
5.  过滤异常线段（角度判断）
6.  最小二乘拟合左右车道线
7.  可视化输出

欢迎 star⭐ 和 fork🍴支持本项目，如有建议或问题请提 issue，我们一起完善它！