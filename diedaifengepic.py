# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
迭代阈值图像分割

迭代法是基于逼近的思想，其步骤如下： 
1． 求出图象的最大灰度值和最小灰度值，分别记为ZMAX和ZMIN，令初始阈值T0=(ZMAX+ZMIN)/2； 
2． 根据阈值TK将图象分割为前景和背景，分别求出两者的平均灰度值ZO和ZB 
3． 求出新阈值TK+1=(ZO+ZB)/2； 
4． 若TK==TK+1，则所得即为阈值；否则转2，迭代计算。

"""
path = 'cat1.jpg'
img = cv2.imread(path)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray_array = np.array(img_gray)

zmax = int(img_gray_array.max())
zmin = int(img_gray_array.min())

t = (zmax + zmin)/2

while True:

    img_zo = np.where(img_gray_array > t, 0, img_gray_array)#大于某个值的元素由0替代
    img_bo = np.where(img_gray_array < t, 0, img_gray_array)#小于某个值的元素由0替代

    zo = np.sum(img_zo)/np.sum(img_zo != 0)
    bo = np.sum(img_bo)/np.sum(img_bo != 0)

    k = (zo + bo)/2

    if abs(t - k) < 0.01:
        break;
    else:
        t = k


#根据最新的阈值进行分割
img_gray_array[img_gray_array > t]  = 255
img_gray_array[img_gray_array <= t] = 0

# plt.imshow(img_gray_array, cmap='gray')
# plt.show()
cv2.imshow('jieguo',img_gray_array)
cv2.waitKey(0)
cv2.imwrite('diedaifenge.jpg',img_gray_array)