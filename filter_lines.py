# @Author : LiZhongzheng
# 开发时间  ：2025-04-29 9:01
import cv2
import numpy as np

"""
剔除出因为误差而被识别出的直线。
如何分解出噪点和车道线那？
    我们可以知道，车道线的斜率大致是相同的，进而可以分辨出噪点和车道线。
"""


def calculate_slope(line):
    """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)


edge_img = cv2.imread('masked_edge_img.jpg', cv2.IMREAD_GRAYSCALE)

# 获取所有线段
lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20)

# 按照斜率分成车道线
left_lines = [line for line in lines if calculate_slope(line) > 0]
right_lines = [line for line in lines if calculate_slope(line) < 0]


def reject_abnormal_lines(lines, threshold):
    """
    剔除斜率不一致的线段
    :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
    """
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)  # 使用 NumPy 的 np.mean 函数计算当前所有斜率的平均值
        diff = [abs(s - mean) for s in slopes]  # 遍历 slopes 列表，计算每个斜率与平均斜率的绝对差值，并将结果存储在 diff 列表中
        idx = np.argmax(diff)  # 使用 NumPy 的 np.argmax 函数找到 diff 列表中最大值的索引，即斜率差异最大的线段。
        if diff[idx] > threshold:  # 如果最大差异大于阈值 threshold，则认为该线段是异常的，将其从 slopes 和 lines 列表中移除。
            slopes.pop(idx)
            lines.pop(idx)
        else:  # 如果最大差异小于或等于阈值，则认为所有线段的斜率已经足够一致，退出循环。
            break
    return lines  # 如果最大差异小于或等于阈值，则认为所有线段的斜率已经足够一致，退出循环。


print('before filter:')
print('left lines number=')
print(len(left_lines))
print('right lines number=')
print(len(right_lines))

reject_abnormal_lines(left_lines, threshold=0.2)
reject_abnormal_lines(right_lines, threshold=0.2)

print('after filter:')
print('left lines number=')
print(len(left_lines))
print('right lines number=')
print(len(right_lines))
