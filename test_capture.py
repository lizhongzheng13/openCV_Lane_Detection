# @Author : LiZhongzheng
# 开发时间  ：2025-04-29 17:06
import cv2

capture = cv2.VideoCapture('video.mp4')
# capture = cv2.VideoCapture(0) #读取当前设备第0个摄像头
while True:
    ret, frame = capture.read()  # ret 视频流的状态，frame 当前帧的图像
    cv2.imshow('frame', frame)
    cv2.waitKey(20)  # 相当于播放速率
