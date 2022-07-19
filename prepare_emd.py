import os
import cv2
from matplotlib.cbook import simple_linear_interpolation 
import numpy as np
from tqdm import tqdm
from pyemd import emd, emd_samples
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
# import ot

def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i, j], i, j])
            count += 1
    return sig

# image to signature for color image
def img_to_sig2(img):
    sig = np.empty((img.size, 3), dtype=np.float32)
    idx = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sig[idx] = np.array([img[i,j], i, j])
            idx += 1
    return sig

def get_signature_from_heatmap(hm):
    nr = hm.shape[0]
    nc = hm.shape[1]
    # print hm

    sig = np.zeros((nr*nc), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            idx = r * nc + c
            sig[idx] = hm[r, c]

    sig[:] /= np.sum(sig[:])
    # print sig
    return sig

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

aug_path = "./augs"
sal_path = "./saliency"

idx = []
xs = []
ys = []

for user in os.listdir(aug_path):
    user = "crj"
    print(user)
    for condition in os.listdir(os.path.join(aug_path, user)):
        if "con" not in condition:
            continue
        print(condition)
        for imgs in tqdm(os.listdir(os.path.join(aug_path, user, condition))):
            # print(imgs)
            img = cv2.imread(os.path.join(aug_path, user, condition, imgs))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            innerpoints = []
            label = 0
            outline = []
            aug_dis = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
            if len(contours) == 0:
                # no augmentation
                pass
            else:
                outline = contours[0]
                if len(contours) > 1:
                    temp = []
                    num = 0
                    for c in contours:
                        if len(c) > num:
                            temp = c
                            num = len(c)
                    outline = temp
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if cv2.pointPolygonTest(outline, (j, i), False) > 0:
                            innerpoints.append([i, j, gray[i][j]])
                
                if len(innerpoints) > 0:
                    innerpoints = np.array(innerpoints)
                    if (np.mean(innerpoints[:, 2])) < 70:
                        print(imgs, np.mean(np.mean(innerpoints[:, 2])))
                        label = 1
                
                    noise_point = np.random.multivariate_normal([np.mean(innerpoints[:, 0]), np.mean(innerpoints[:, 1])], [[np.std(innerpoints[:, 0]), 0], [0, np.std(innerpoints[:, 1])]], 1000)
                    for point in noise_point:
                        aug_dis[int(point[0]), int(point[1])] += 1 / 1000 * 255

            sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            
            sal_img_resize = cv2.resize(sal_img, (38, 22), interpolation=cv2.INTER_LANCZOS4)
            sal_img_resize /= np.sum(sal_img_resize)
            aug_resize = cv2.resize(aug_dis, (38, 22), interpolation=cv2.INTER_LANCZOS4)
            aug_resize /= np.sum(aug_resize)

            emd = sliced_wasserstein(sal_img_resize, aug_resize, 100)

            if len(innerpoints) == 0 or np.isnan(emd):
                pass
            else:
                # xs.append(emd_samples(binary, sal_img))
                xs.append(emd)
                idx.append(imgs)
                ys.append(label)

            # cv2.imshow("test", aug_dis)
            # cv2.waitKey(0)

            # print(sliced_wasserstein(sal_img_resize, aug_resize, 100))
            # print(emd_samples(sal_img_resize, aug_resize))
            # print(np.sum(sal_img_resize), np.sum(aug_resize))
            # exit()

            # print(sliced_wasserstein(sal_img_resize, binary_resize, 100))
            # print(np.sum(sal_img_resize))
            # print(sliced_wasserstein(sal_img_resize, binary_resize, 100) / np.sum(sal_img_resize))
            # cv2.imshow("test", sal_img)
            # cv2.waitKey()
            # cv2.imshow("test", sal_img_resize)
            # cv2.waitKey()
            # exit()
            # print(sliced_wasserstein(sal_img_resize, binary_resize, 100))
            # print(np.sum(sal_img_resize))
            # exit()
        df = pd.DataFrame({"name": idx, "emd": xs, "label": ys})
        df.to_csv("./data_test.csv")
        # exit()
    break  
                
