# @Author : LiZhongzheng
# 开发时间  ：2025-04-29 8:41
import cv2
import numpy as np

edge_img = cv2.imread('edge_img.jpg', cv2.IMREAD_GRAYSCALE)

mask = np.zeros_like(edge_img)
mask = cv2.fillPoly(mask, np.array([[[0, 368], [300, 210], [340, 210], [640, 368]]]),
                    color=255)  # 顺序：左下，左上，右上，右下
masked_edge_img = cv2.bitwise_and(edge_img, mask)
cv2.imwrite('masked_edge_img.jpg', masked_edge_img)
cv2.imshow('masked', masked_edge_img)
cv2.waitKey(0)
