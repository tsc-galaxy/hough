import numpy as np
import cv2
from PIL import Image, ImageEnhance


def img_processing(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)#阈值变换，cv2.THRESH_OTSU适合用于双峰值图像,ret是阈值，binary是变换后的图像
    # canny边缘检测
    edges = cv2.Canny(binary, ret-30, ret+30, apertureSize=3)#图像，最小阈值，最大阈值，sobel算子的大小
    return edges


def line_detect(img):
    img = Image.open(img)

    img = ImageEnhance.Contrast(img).enhance(3)#对比度增强类,用于调整图像的对比度,3为增强3倍
    img = np.array(img)
    print(img.shape)
    result = img_processing(img)#返回来的是一个矩阵
    print(result.shape)
    # 霍夫线检测
    lines = cv2.HoughLinesP(result, 1, 1 * np.pi / 180, 10, minLineLength=10, maxLineGap=5)#统计概率霍夫线变换函数：图像矩阵，极坐标两个参数，一条直线所需最少的曲线交点，组成一条直线的最少点的数量，被认为在一条直线上的亮点的最大距离
    print("Line Num : ", len(lines))

    # 画出检测的线段
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 120, 0),2)
        pass
    img = Image.fromarray(img, 'RGB')
    img.show()


if __name__ == "__main__":
    line_detect("zhongzhi.jpg")
    pass