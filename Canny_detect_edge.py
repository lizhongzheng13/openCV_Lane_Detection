# @Author : LiZhongzheng
# 开发时间  ：2025-04-29 8:30
import cv2

img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

edge_img = cv2.Canny(img, 50, 100)

cv2.imshow('edge_img', edge_img)
cv2.waitKey(0)
cv2.imwrite('edge_img.jpg', edge_img)
