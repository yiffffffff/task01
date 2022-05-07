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
from matplotlib import pyplot as plt

#这个库不是必须的，用来提供进度条
from tqdm import trange
import cv2

SEAM_COLOR = np.array([255, 200, 200])
def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    r, c, _ = vis.shape
    if boolmask is not None:
        for i in range(r):
            vis[i][boolmask[i]] = SEAM_COLOR
    if rotate:
        vis = np.rot90(vis, 3, (0, 1))
    cv2.imshow("visualization", vis)
    cv2.waitKey(30)
    return vis


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
        img=carve_column(img)

    return img

def add_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    img=add_column(img,new_c-c)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def add_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = add_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def carve_column(img):
    r,c,channels=img.shape
    M, backtrack = minimumEnergySeam(img)
    # 创建一个(r,c)矩阵，填充值为True
    # 后面会从值为False的图像中移除所有像素
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(M[-1])
    seam=np.ones(r,dtype=np.int)
    for i in reversed(range(r)):
        # 标记出需要删除的像素
        seam[i]=j
        mask[i, j] = False
        j=backtrack[i, j]
    visualize(img, seam, 0)
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def add_seam(im, seam_idx):
    h, w = im.shape[:2]
    visualize(im, seam_idx, 0)
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output

def add_column(img,add_c):
    r,c,channels=img.shape
    M, backtrack = minimumEnergySeam(img)
    seams_record = np.ones((add_c,r),dtype=np.int)
    jmax=np.argmax(M[-1])
    for i in range(add_c):
        j = np.argmin(M[-1])
        M[-1,j]=M[-1,jmax]
        seam=np.ones(r,dtype=np.int)
        for i2 in reversed(range(r)):
            seam[i2]=j
            j=backtrack[i2,j]
        seams_record[i]=seam
    for i in trange(add_c):
        img=add_seam(img,seams_record[i])
        for i2 in range(i+1,add_c):
            seams_record[i2][np.where(seams_record[i2]>=seams_record[i])]+=1

    img = add_seam(img, seam)
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

def main():
    if len(sys.argv) != 5:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]

    img = imread(in_filename).astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if scale<1:
        if which_axis == 'r':
            out = crop_r(img, scale)
        elif which_axis == 'c':
            out = crop_c(img, scale)
        else:
            print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
            sys.exit(1)
    elif scale>1:
        if which_axis == 'r':
            out = add_r(img, scale)
        elif which_axis == 'c':
            out = add_c(img, scale)
        else:
            print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
            sys.exit(1)
            
    out = out.astype(np.uint8) 
    out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
    imwrite(out_filename, out.astype(np.uint8))

if __name__ == '__main__':
    main()