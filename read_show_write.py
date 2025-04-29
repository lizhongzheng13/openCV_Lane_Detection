# @Author : LiZhongzheng
# 开发时间  ：2025-04-29 8:28
import cv2

img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)  # 转为灰度图
cv2.imshow('img', img)
cv2.waitKey(0)

cv2.imwrite('img_gray.jpg', img)
