'''
Usage: python seam_carving.py <r/c> <scale> <image_in> <image_out> <energy_map_type>
<r/c> 删除列还是删除行
<scale> 缩小为原来的多少(0,1)之间的浮点数
<energy_map_type> c:calc_energy b:backward_energy f:forward_energy
'''
import sys
from imageio import imread,imwrite
from scipy.ndimage import convolve
import cv2
import numba
import numpy as np
from tqdm import trange
from scipy import ndimage as ndi


SEAM_COLOR = np.array([255, 200, 200])
def visualize(im, seams, rotate=False):
    vis = im.astype(np.uint8)
    r, c, _ = vis.shape
    if seams is not None:
        for i in range(r):
            vis[i][seams[i]] = SEAM_COLOR
    if rotate:
        vis = np.rot90(vis, 3, (0, 1))
    cv2.imshow("visualization", vis)
    cv2.waitKey(30)
    return vis


@numba.jit()
def cumulative_energy(energy):
    r, c = energy.shape
    backtrack = np.zeros((r, c), dtype=np.int64)
    energy_map = energy.copy()
    energy_map[0] = energy[0]

    for i in range(1,r):
        for j in range(0,c):
            #处理左侧边缘
            if j==0:
                idx=np.argmin(energy_map[i-1,j:j+2])
                backtrack[i,j]=idx
                min_energy=energy_map[i-1,idx+j]
            #处理右侧边缘
            elif j==c-1:
                idx=np.argmin(energy_map[i - 1, j - 1:j + 1])
                backtrack[i, j] = idx-1 #idx + j - 1
                min_energy = energy_map[i - 1, idx + j - 1]
            else:
                idx=np.argmin(energy_map[i-1,j-1:j+2])
                backtrack[i,j]=idx-1 #idx+j-1
                min_energy=energy_map[i-1,idx+j-1]
            energy_map[i,j]+=min_energy

    return backtrack,energy_map


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


def backward_energy(im):
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    return grad_mag


def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
      
    return energy


def crop_c(img, scale_c,energy_map_type,rotated):
    r, c, _ = img.shape
    new_c = int((1-scale_c) * c)
    img=resize_image(img,new_c,energy_map_type,rotated)
    return img


def crop_r(img, scale_r,energy_map_type):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r,energy_map_type,True)
    img = np.rot90(img, 3, (0, 1))
    return img


def add_c(img, scale_c,energy_map_type,rotated):
    r, c, _ = img.shape
    new_c = int(scale_c * c)
    img=add_column(img,new_c-c,energy_map_type,rotated)
    return img


def add_r(img, scale_r,energy_map_type):
    img = np.rot90(img, 1, (0, 1))
    img = add_c(img, scale_r,energy_map_type,True)
    img = np.rot90(img, 3, (0, 1))
    return img


def find_seam(paths, end_x):
    r,c=paths.shape[:2]
    seam = [end_x]

    for i in range(r-1, 0, -1):
        cur_x = seam[-1]
        offset_of_prev_x = paths[i][cur_x]
        seam.append(cur_x+offset_of_prev_x)
    seam.reverse()

    return seam


def remove_seam(img, seam):
    r,c=img.shape[:2]
    img_removed=np.zeros((r,c-1,3))

    for r2 in range(r):
        img_removed[r2]=np.delete(img[r2], seam[r2], axis=0)

    return img_removed
  
  
def add_seam(im, seam_idx,rotated=False):
    h, w = im.shape[:2]
    visualize(im, seam_idx, rotated)
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

def add_column(img,add_c,energy_map_type,rotated=False):
    img_original=img.copy()
    r,c,channels=img.shape
    seams_record = np.ones((add_c,r),dtype=int)

    for i in trange(add_c, desc='cropping image by {0} pixels'.format(add_c)):

        if energy_map_type == 'c':
            energy_map=calc_energy(img)
        elif energy_map_type == 'b':
            energy_map=backward_energy(img)
        else:
            energy_map=forward_energy(img)

        backtrack, energy_totals = cumulative_energy(energy_map)
        energy_minc=list(energy_totals[-1]).index(min(energy_totals[-1]))
        seam = find_seam(backtrack, energy_minc)
        visualize(img, seam, rotated)
        seams_record[i]=seam
        img = remove_seam(img, seam)

    for i in trange(add_c, desc='expanding image by {0} pixels'.format(add_c)):
        img_original=add_seam(img_original,seams_record[i],rotated)
        for i2 in range(i+1,add_c):
            seams_record[i2][np.where(seams_record[i2]>=seams_record[i])]+=2

    return img_original


def resize_image(full_img, cropped_c,energy_map_type,rotated=False,):
    img = full_img.copy()

    for _ in trange(cropped_c, desc='cropping image by {0} pixels'.format(cropped_c)):

        if energy_map_type == 'c':
            energy_map=calc_energy(img)
        elif energy_map_type == 'b':
            energy_map=backward_energy(img)
        else:
            energy_map=forward_energy(img)

        backtrack, energy_totals = cumulative_energy(energy_map)
        energy_minc=list(energy_totals[-1]).index(min(energy_totals[-1]))
        seam = find_seam(backtrack, energy_minc)
        visualize(img, seam, rotated)
        img = remove_seam(img, seam)

    return img


def main():
    if len(sys.argv) != 6:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]
    energy_map_type = sys.argv[5]

    img = imread(in_filename).astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if scale<1:
        if which_axis == 'r':
            out = crop_r(img, scale, energy_map_type)
        elif which_axis == 'c':
            out = crop_c(img, scale, energy_map_type, False)
        else:
            print('usage: carver.py <r/c> <scale> <image_in> <image_out>2', file=sys.stderr)
            sys.exit(1)

    elif scale>1:
        if which_axis == 'r':
            out = add_r(img, scale, energy_map_type)
        elif which_axis == 'c':
            out = add_c(img, scale, energy_map_type, False)
        else:
            print('usage: carver.py <r/c> <scale> <image_in> <image_out>3', file=sys.stderr)
            sys.exit(1)
            
    out = out.astype(np.uint8) 
    out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
    imwrite(out_filename, out.astype(np.uint8))


if __name__ == "__main__":
    main()
