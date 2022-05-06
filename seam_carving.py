'''
Usage: python seam_carving.py <r/c> <scale> <image_in> <image_out>
<r/c> 删除列还是删除行
<scale> 缩小为原来的多少(0,1)之间的浮点数

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
    #stack函数,我们要对红绿蓝都计算
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


def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    #这里用了显示进度条的库
    for i in trange(c - new_c): 
        img = carve_column(img)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def carve_column(img):
    r,c,channels=img.shape
    M, backtrack = minimumEnergySeam(img)
    # 创建一个(r,c)矩阵，填充值为True
    # 后面会从值为False的图像中移除所有像素
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        # 标记出需要删除的像素
        mask[i, j] = False
        j = backtrack[i, j]
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img


def minimumEnergySeam(img):
    #shape 能返回图片的宽高和通道数，
    r,c,channels=img.shape
    energy_map=calc_energy(img)

    M=energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int) #创建一个同大小，但初始化为0的数组

    for i in range(1,r):
        for j in range(0,c):
            #处理左侧边缘
            if j==0:
                idx=np.argmin(M[i-1,j:j+2])
                backtrack[i,j]=idx+j
                min_energy=M[i-1,idx+j]
            #处理右侧边缘
            elif j==c-1:
                idx=idx = np.argmin(M[i - 1, j - 1:j + 1])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            else:
                idx=np.argmin(M[i-1,j-1:j+2])
                backtrack[i,j]=idx+j-1
                min_energy=M[i-1,idx+j-1]
            M[i,j]+=min_energy
    return M,backtrack


def minimumEnergySeam2(img):
    #shape 能返回图片的宽高和通道数，
    r,c,channels=img.shape
    energy_map=calc_energy(img)

    M=energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int) #创建一个同大小，但初始化为0的数组

    for i in range(1,r):
        for j in range(0,c):
            #处理左侧边缘
            if j==0:
                idx=np.argmin(M[i-1,j:j+2])
                backtrack[i,j]=idx+j
                min_energy=M[i-1,idx+j]
            #处理右侧边缘
            elif j==c-1:
                idx=idx = np.argmin(M[i - 1, j - 1:j + 1])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            else:
                idx=np.argmin(M[i-1,j-1:j+2])
                backtrack[i,j]=idx+j-1
                min_energy=M[i-1,idx+j-1]
            M[i,j]+=min_energy
    return M,backtrack



def main():
    if len(sys.argv) != 5:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]

    img = imread(in_filename)

    if which_axis == 'r':
        out = crop_r(img, scale)
    elif which_axis == 'c':
        out = crop_c(img, scale)
    else:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()