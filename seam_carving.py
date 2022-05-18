"""
Usage: python seam_carving.py <rs> <r/c> <scale> <image_in> <image_out> <energy_map_type> <y/n>
Or:python seam_carving.py <rm> <image_in> <image_out> <image_mask>
<rm/rs> rm represents remove rs represents resize
<r/c> 删除列还是删除行
<scale> 缩小为原来的多少(0,1)之间的浮点数
<energy_map_type> c:calc_energy b:backward_energy f:forward_energy
<y/n> y:使用grabCut n:不使用grabCut
"""
import sys
from scipy.ndimage import convolve
import cv2
import numba
import numpy as np
from tqdm import trange
from scipy import ndimage as ndi

ENERGY_MASK_CONST = 100000.0              # large energy value for protective masking
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask
SEAM_COLOR = np.array([255, 200, 200])


def visualize(im, seams, rotate=False):
    vis = im.astype(np.uint8)
    r, c, _ = vis.shape
    if seams is not None:
        for i in range(r):
            vis[i][seams[i]] = SEAM_COLOR
    if rotate:
        vis = np.rot90(vis, 3, (0, 1))
    #cv2.namedWindow('visualization', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("visualization", 500, 500)
    cv2.imshow("visualization", vis)
    cv2.waitKey(30)
    return vis


@numba.jit()
def cumulative_energy(energy):
    r, c = energy.shape
    backtrack = np.zeros((r, c), dtype=np.int64)
    energy_map = energy.copy()
    energy_map[0] = energy[0]

    for i in range(1, r):
        for j in range(0, c):
            # 处理左侧边缘
            if j == 0:
                idx = np.argmin(energy_map[i - 1, j : j + 2])
                backtrack[i, j] = idx
                min_energy = energy_map[i - 1, idx + j]
            # 处理右侧边缘
            elif j == c - 1:
                idx = np.argmin(energy_map[i - 1, j - 1 : j + 1])
                backtrack[i, j] = idx - 1  # idx + j - 1
                min_energy = energy_map[i - 1, idx + j - 1]
            else:
                idx = np.argmin(energy_map[i - 1, j - 1 : j + 2])
                backtrack[i, j] = idx - 1  # idx+j-1
                min_energy = energy_map[i - 1, idx + j - 1]
            energy_map[i, j] += min_energy

    return backtrack, energy_map

def calc_energy(img):
    # np.array函数创建了一个类似矩阵的东西
    # 一个二维数组
    filter_x = np.array(
        [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ]
    )
    # stack函数,我们要对红绿蓝都计算
    filter_x = np.stack([filter_x] * 3, axis=2)

    filter_y = np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ]
    )
    filter_y = np.stack([filter_y] * 3, axis=2)

    # 类型转化，转化为float
    img = img.astype("float32")

    # 计算卷积和
    convolved = np.absolute(convolve(img, filter_x)) + np.absolute(
        convolve(img, filter_y)
    )
    # 将红绿蓝通道中的能量相加
    energy_map = convolved.sum(axis=2)

    return energy_map


def backward_energy(im):
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode="wrap")
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode="wrap")
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
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR
        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    return energy


def crop_c(img, scale_c, energy_map_type, if_grabCut, rotated):
    r, c, _ = img.shape
    new_c = int((1 - scale_c) * c)
    img = resize_image(img, new_c, energy_map_type, if_grabCut, rotated)
    return img


def crop_r(img, scale_r, energy_map_type, if_grabCut):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r, energy_map_type, if_grabCut, True)
    img = np.rot90(img, 3, (0, 1))
    return img


def add_c(img, scale_c, energy_map_type, if_grabCut, rotated):
    r, c, _ = img.shape
    new_c = int(scale_c * c)
    img = add_column(img, new_c - c, energy_map_type, if_grabCut, rotated)
    return img


def add_r(img, scale_r, energy_map_type, if_grabCut):
    img = np.rot90(img, 1, (0, 1))
    img = add_c(img, scale_r, energy_map_type, if_grabCut, True)
    img = np.rot90(img, 3, (0, 1))
    return img


def find_seam(paths, end_x):
    r, c = paths.shape[:2]
    seam = [end_x]
    for i in range(r - 1, 0, -1):
        cur_x = seam[-1]
        offset_of_prev_x = paths[i][cur_x]
        seam.append(cur_x + offset_of_prev_x)
    seam.reverse()

    return seam

def remove_seam(img, seam):
    r, c = img.shape[:2]
    img_removed = np.zeros((r, c - 1, 3))
    for r2 in range(r):
        img_removed[r2] = np.delete(img[r2], seam[r2], axis=0)
    return img_removed

def remove_seam_grayscale(img ,seam):
    r, c = img.shape[:2]
    img_removed = np.zeros((r, c - 1))
    for i in range(r):
        img_removed[i]=np.delete(img[i], seam[i], axis=0)
    return img_removed

def add_seam(im, seam_idx, rotated=False):
    h, w = im.shape[:2]
    visualize(im, seam_idx, rotated)
    output = np.zeros((h, w + 1, 3))

    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col : col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1 :, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1 : col + 1, ch])
                output[row, :col, ch] = im[row, :col, ch]
                output[row, col, ch] = p
                output[row, col + 1 :, ch] = im[row, col:, ch]

    return output

def add_column(img, add_c, energy_map_type, if_grabCut, rotated=False):
    img_original = img.copy()
    r, c, channels = img.shape
    seams_record = np.ones((add_c, r), dtype=int)

    if(if_grabCut):
        energy_grabCut=improve_seam(img,rotated)

    for i in trange(add_c, desc="cropping image by {0} pixels".format(add_c)):
        if energy_map_type == "c":
            energy_map = calc_energy(img)
        elif energy_map_type == "b":
            energy_map = backward_energy(img)
        else:
            energy_map = forward_energy(img)

        if(if_grabCut):
            energy_map+=energy_grabCut

        backtrack, energy_totals = cumulative_energy(energy_map)
        energy_minc = list(energy_totals[-1]).index(min(energy_totals[-1]))
        seam = find_seam(backtrack, energy_minc)
        visualize(img, seam, rotated)
        seams_record[i] = seam
        img = remove_seam(img, seam)
        if(if_grabCut):
            energy_grabCut=remove_seam_grayscale(energy_grabCut,seam)

    for i in trange(add_c, desc="expanding image by {0} pixels".format(add_c)):
        img_original = add_seam(img_original, seams_record[i], rotated)
        for i2 in range(i + 1, add_c):
            seams_record[i2][np.where(seams_record[i2] >= seams_record[i])] += 2

    return img_original

def resize_image(
    full_img,
    cropped_c,
    energy_map_type,
    if_grabCut,
    rotated=False,
):
    img = full_img.copy()
    if(if_grabCut):
        energy_grabCut=improve_seam(img,rotated)

    for _ in trange(cropped_c, desc="cropping image by {0} pixels".format(cropped_c)):

        if energy_map_type == "c":
            energy_map = calc_energy(img)
        elif energy_map_type == "b":
            energy_map = backward_energy(img)
        else:
            energy_map = forward_energy(img)

        if(if_grabCut):
            energy_map+=energy_grabCut

        backtrack, energy_totals = cumulative_energy(energy_map)
        energy_minc = list(energy_totals[-1]).index(min(energy_totals[-1]))
        seam = find_seam(backtrack, energy_minc)
        visualize(img, seam, rotated)
        img = remove_seam(img, seam)
        if(if_grabCut):
            energy_grabCut=remove_seam_grayscale(energy_grabCut,seam)

    return img


def improve_seam(im_rgb, rotated):
    im_cut = im_rgb.copy().astype('uint8')
    if rotated:
        im_cut_rot = np.rot90(im_cut, 3, (0, 1))
    else:
        im_cut_rot = im_cut.copy()
    r0, c0 = im_cut.shape[:2]
    r = cv2.selectROI('input', im_cut_rot, False) 
    rect = (int(r[1]), int(r0-r[0]-r[2]), int(r[3]), int(r[2]))
    rows, cols = im_cut.shape[0], im_cut.shape[1]
    mask = np.zeros(im_cut.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(im_cut,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')#0和2做背景
    energy_map =calc_energy(im_rgb)
    for i in range(rows):
            for j in range(cols):
                if (mask2[i][j] == 1):
                    energy_map[i][j] = 10000
                elif ( mask2[i][j]==3):
                    energy_map[i][j] = 10000
                else:
                    energy_map[i][j] = 0
    return energy_map

def object_removal(img,mask):
    h, w = img.shape[:2]
    output = img
    
    while len(np.where(mask>MASK_THRESHOLD)[0])> 0 :
        energy_map=calc_energy(img)
        #在能量图中导入MASK的影响
        energy_map[np.where(mask>MASK_THRESHOLD)]= -ENERGY_MASK_CONST
        backtrack, energy_totals = cumulative_energy(energy_map)
        energy_minc = list(energy_totals[-1]).index(min(energy_totals[-1]))
        seam = find_seam(backtrack, energy_minc)
        visualize(img, seam)
        img = remove_seam(img, seam)
        mask=remove_seam_grayscale(mask,seam)
    num_add=int(w-img.shape[1])
    output=add_column(img,num_add,'c')
    return output

def main():
    if len(sys.argv) == 5:
        in_filename = sys.argv[2]
        out_filename = sys.argv[3]
        mask_filename = sys.argv[4]
        img_in = cv2.imread(in_filename).astype(np.float64)
        #read in protect mask in gray scale
        img_mask = cv2.imread(mask_filename,0).astype(np.float64)
        img_out=object_removal(img_in,img_mask)
        cv2.imwrite(out_filename, img_out.astype(np.float64))
    elif len(sys.argv) == 8:

        which_axis = sys.argv[2]
        scale = float(sys.argv[3])
        in_filename = sys.argv[4]
        out_filename = sys.argv[5]
        energy_map_type = sys.argv[6]
        if_grabCut =sys.argv[7]

        img = cv2.imread(in_filename).astype(np.float64)
        if(if_grabCut)=='y':
            have_grabCut=True
        else:
            have_grabCut=False

        if scale < 1:
            if which_axis == "r":
                out = crop_r(img, scale, energy_map_type, have_grabCut)
            elif which_axis == "c":
                out = crop_c(img, scale, energy_map_type, have_grabCut, False)
            else:
                print(
                    "Usage: python seam_carving.py <rm/rs> <r/c> <scale> <image_in> <image_out> <energy_map_type> <y/n>",
                    file=sys.stderr,
                )
                sys.exit(1)

        elif scale > 1:
            if which_axis == "r":
                out = add_r(img, scale, energy_map_type, have_grabCut)
            elif which_axis == "c":
                out = add_c(img, scale, energy_map_type, have_grabCut, False)
            else:
                print(
                    "Usage: python seam_carving.py <rm/rs> <r/c> <scale> <image_in> <image_out> <energy_map_type> <y/n>",
                    file=sys.stderr,
                )
                sys.exit(1)

        out = out.astype(np.float64)
        cv2.imwrite(out_filename, out.astype(np.float64))
    else:
        print(
            "Usage: python seam_carving.py <rm/rs> <r/c> <scale> <image_in> <image_out> <energy_map_type> <y/n>",
            file=sys.stderr,
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
