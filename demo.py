import os
import cv2
from matplotlib.cbook import simple_linear_interpolation 
import numpy as np
from tqdm import tqdm
from pyemd import emd, emd_samples
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter

aug_path = "./augs"
sal_path = "./saliency"

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

for user in os.listdir(aug_path):
    user = "crj"
    print(user)
    for condition in os.listdir(os.path.join(aug_path, user)):
        if "con" not in condition:
            continue
        print(condition)
        for imgs in tqdm(os.listdir(os.path.join(aug_path, user, condition))):
            imgs = "3621.png"
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

            aug_blurred = blur_image(gray)
            sal_img_resize = cv2.resize(sal_img, (38, 22), interpolation=cv2.INTER_LANCZOS4)
            # sal_img_resize /= np.sum(sal_img_resize)
            aug_resize = cv2.resize(aug_dis, (38, 22), interpolation=cv2.INTER_LANCZOS4)
            # aug_resize /= np.sum(aug_resize)

            print(np.sum(aug_dis))
            print(np.sum(aug_resize))
            print(np.sum(sal_img))
            print(np.sum(sal_img_resize))
            sal_flat = get_signature_from_heatmap(sal_img_resize)
            aug_flat = get_signature_from_heatmap(aug_resize)
            print(np.sum(sal_flat[:, 0]))
            emd, lowerbound, flow_matrix = cv2.EMD(sal_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)

            print(emd)
            exit()

            if label == 0:
                print(imgs)
                print(np.sum(sal_flat[:, 0]))
                print(emd / np.sum(sal_flat[:, 0]))
                cv2.imwrite("saliency.png", sal_img)
                cv2.imwrite("saliency_resize.png", sal_img_resize)
                cv2.imwrite("aug.png", aug_dis)
                cv2.imwrite("aug_resize.png", aug_resize)
                cv2.imshow("saliency", sal_img)
                cv2.waitKey(0)
                cv2.imshow("saliency", sal_img_resize)
                cv2.waitKey(0)
                cv2.imshow("augmentation", aug_dis)
                cv2.waitKey(0)
                cv2.imshow("augmentation", aug_resize)
                cv2.waitKey(0)

            exit()

            
            