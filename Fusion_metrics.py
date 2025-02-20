import numpy as np
import torch
import math
import cv2


def x_CC_function(ref, tar):
    "Pearson coefficient, cui"
    # Get dimensions
    batch, bands, rows, cols = tar.shape
    tar = tar.detach().cpu().numpy()
    ref = ref.detach().cpu().numpy()
    # Initialize output array
    out = np.zeros((batch, bands))
    for b in range(batch):
        # Compute cross correlation for each band
        for i in range(bands):
            tar_tmp = tar[b, i, :, :]
            ref_tmp = ref[b, i, :, :]
            cc = np.corrcoef(tar_tmp.flatten(), ref_tmp.flatten())
            out[b, i] = cc[0, 1]


def CC_function(A, F):
    "Cosine similarity"
    cc = []
    for i in range(A.shape[0]):
        cc.append(np.sum((A[i] - np.mean(A[i])) * (F[i] - np.mean(F[i]))) / np.sqrt(
            np.sum((A[i] - np.mean(A[i])) ** 2) * np.sum((F[i] - np.mean(F[i])) ** 2)))
    return np.mean(cc)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def SAM(x_true, x_pred):
    """calculate SAM method in code"""
    batch = x_true.shape[0]
    sam = np.array([])
    dot_sum = np.sum(x_true * x_pred, axis=1)
    norm_true = np.linalg.norm(x_true, axis=1)
    norm_pred = np.linalg.norm(x_pred, axis=1)
    res = np.arccos(dot_sum / norm_pred / norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0
    return np.mean(res)


def ssim1(img1, img2):
    """计算结构相似性指数(SSIM)"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_values = []  # 存储每个图像样本的 SSIM 值

    for i in range(img1.shape[0]):  # 遍历每个图像样本
        single_img1 = img1[i].astype(np.float64)
        single_img2 = img2[i].astype(np.float64)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(single_img1, -1, window)
        mu2 = cv2.filter2D(single_img2, -1, window)
        mu1 = mu1[3:-3, 3:-3]
        mu2 = mu2[3:-3, 3:-3]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(single_img1 ** 2, -1, window)[3:-3, 3:-3] - mu1_sq
        sigma2_sq = cv2.filter2D(single_img2 ** 2, -1, window)[3:-3, 3:-3] - mu2_sq
        sigma12 = cv2.filter2D(single_img1 * single_img2, -1, window)[3:-3, 3:-3] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        ssim_values.append(ssim_map.mean())  # 存储每个图像样本的 SSIM 值

    return np.mean(ssim_values)  # 返回所有图像样本的 SSIM 平均值