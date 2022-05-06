'''
导出能量图
Usage: python cal_energy.py  <image_in> <image_out>
'''
import sys
import os
import time
#数学运算库
import numpy as np
from imageio import imread,imwrite
from scipy.ndimage import convolve

#这个库不是必须的，用来提供进度条
from tqdm import trange

#能量函数
#使用了sobel算子，Google下sobel算子就知道大概过程了
def calc_energy(img):
    #np.array函数创建了一个类似矩阵的东西
    #一个二维数组
    filter_x = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    #stack函数
    filter_x = np.stack([filter_x] * 3, axis=2)

    filter_y = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_y = np.stack([filter_y] * 3, axis=2)

    #类型转化，转化为float
    img = img.astype('float32')

    #计算卷积和
    convolved = np.absolute(convolve(img, filter_x)) + np.absolute(convolve(img, filter_y))
    #将红绿蓝通道中的能量相加
    energy_map = convolved.sum(axis=2)

    return energy_map

def main():
    in_filename = sys.argv[1]
    out_filename = sys.argv[2]

    img = imread(in_filename)
    out = calc_energy(img)
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()