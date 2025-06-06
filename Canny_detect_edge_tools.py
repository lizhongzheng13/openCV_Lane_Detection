# @Author : LiZhongzheng
# 开发时间  ：2025-04-29 8:33
import cv2

cv2.namedWindow('edge_detection')
cv2.createTrackbar('minThreshold', 'edge_detection', 50, 1000, lambda x: x)  # 创建滑动条（trackbar）的函数
cv2.createTrackbar('maxThreshold', 'edge_detection', 100, 1000, lambda x: x)
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

while True:
    minThreshold = cv2.getTrackbarPos('minThreshold', 'edge_detection')
    maxThreshold = cv2.getTrackbarPos('maxThreshold', 'edge_detection')
    edges = cv2.Canny(img, minThreshold, maxThreshold)
    cv2.imshow('edge_detection', edges)
    cv2.waitKey(10)
