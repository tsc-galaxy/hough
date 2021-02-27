import cv2 as cv
import numpy as np
import math


# Otsu算法使用的是聚类的思想：
# ①把图像的灰度数按灰度级分成2个部分，使得两个部分之间的灰度值差异最大，每个部分之间的灰度差异最小；
# ②通过方差的计算来寻找一个合适的灰度级别来划分；
def calc_grayhist(image):
    #图像宽高
    rows,cols=image.shape
    grayhist=np.zeros([256],np.uint64)
    for i in range(rows):
        for j in range(cols):
            grayhist[image[i][j]]+=1
    return grayhist

def OTSU(image):
    rows, cols = image.shape
    grayhist=calc_grayhist(image)
    #归一化直方图
    uniformgrayhist=grayhist/float(rows*cols)
    #计算零阶累积矩和一阶累积矩
    zeroaccumulat = np.zeros([256],np.float32)
    oneaccumulat = np.zeros([256], np.float32)
    for k in range(256):
        if k==0:
            zeroaccumulat[k]=uniformgrayhist[0]
            oneaccumulat[k]=k*uniformgrayhist[0]
        else:
            zeroaccumulat[k]=zeroaccumulat[k-1]+uniformgrayhist[k]
            oneaccumulat[k]=oneaccumulat[k-1]+k*uniformgrayhist[k]
    #计算间类方差
    variance=np.zeros([256],np.float32)
    for k in range(256):
        if zeroaccumulat[k]==0 or zeroaccumulat[k]==1:
            variance[k]=0
        else:
            variance[k]=math.pow(oneaccumulat[255]*zeroaccumulat[k]-
                  oneaccumulat[k],2)/(zeroaccumulat[k]*(1.0-zeroaccumulat[k]))
    threshLoc=np.where(variance[0:255]==np.max(variance[0:255]))
    thresh=threshLoc[0][0]
    #阈值分割
    threshold=np.copy(image)
    threshold[threshold>thresh]=255
    threshold[threshold <= thresh] = 0
    return threshold

if __name__=="__main__":
    img=cv.imread("cat1.jpg")
    gray_dst = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    Otsu_dst=OTSU(gray_dst)
    cv.imshow("gray dst",gray_dst)
    cv.imshow("Otsu dst", Otsu_dst)
    cv.imwrite("ostufenge.jpg", Otsu_dst)
    cv.waitKey(0)
    cv.destroyAllWindows()