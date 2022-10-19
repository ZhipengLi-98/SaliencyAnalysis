import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
import threading
from scipy.ndimage import gaussian_filter

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

    # sig[:, 0] /= np.sum(sig[:, 0])
    # print sig
    return sig

if __name__ == "__main__":
    imgs_path = "./formal/imgs"
    saliency_path = "./formal/saliency"
    for user in os.listdir(imgs_path):
        user = "test"
        print(user)
        for condition in os.listdir(os.path.join(imgs_path, user)):
            condition = "test_pos_video_virtual2"
            print(condition)
            aug_path = os.path.join(imgs_path, user, condition, condition)
            for folder in os.listdir(os.path.join(imgs_path, user, condition)):
                if folder.split("_")[-1].split(".")[0] == "ani":
                    aug_path = os.path.join(imgs_path, user, condition, folder)
                    input_path = os.path.join(imgs_path, user, condition, folder)
                    output_path = os.path.join(saliency_path, user, folder)
                    main(input_path, output_path)
            break
        break