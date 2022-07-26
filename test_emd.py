import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter
import math

def blur_image(image):
    blurred_img = gaussian_filter(image, sigma=7)  # sigma=7,

    min_img = np.min(blurred_img)
    max_img = np.max(blurred_img)

    norm_value = max_img - min_img

    if max_img > 0 and norm_value > 0:
        blurred_img = (blurred_img - min_img) / (norm_value) * 255.

    # print(str(min_img) + ' ' + str(max_img) + ' ' + str(np.min(blurred_img)) + '  ' + str(np.max(blurred_img)))

        # blurred_img = gaussian_filter(map, sigma=2)  # sigma=7,
        # blurred_img2 = gaussian_filter(map, sigma=10)
        # blurred_img = cv2.addWeighted(blurred_img, 0.2, blurred_img2, 1.5, 0)
    return blurred_img

def sliced_wasserstein(X, Y, num_proj):
    dim = X.shape[1]
    ests = []
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein
        ests.append(wasserstein_distance(X_proj, Y_proj))
    return np.mean(ests)

def get_signature_from_heatmap(hm):
    nr = hm.shape[0]
    nc = hm.shape[1]
    # print hm

    sig = np.zeros((nr * nc, 3), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            idx = r * nc + c
            sig[idx, 0] = max(hm[r, c], 0)
            sig[idx, 1] = r
            sig[idx, 2] = c

    sum_hm = np.sum(sig[:, 0])

    if sum_hm < 0.0001:
        return None

    sig[:, 0] /= np.sum(sig[:, 0])
    # print sig
    return sig

first = np.zeros((3, 3))
first[0, 0] = 1
first[2, 2] = 1
second = np.zeros((3, 3))
# second[75, 43, 0] = 0.1
second[1, 1] = 1

print(sliced_wasserstein(first, second, 100))

first_sig = get_signature_from_heatmap(first)
second_sig = get_signature_from_heatmap(second)

print(first_sig)
print(second_sig)

print(cv2.EMD(first_sig, second_sig, distType=cv2.DIST_L2, lowerBound=0)[0])
