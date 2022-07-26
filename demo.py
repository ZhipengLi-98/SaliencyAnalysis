import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter

aug_path = "./augs"
sal_path = "./saliency"

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

idx = []
xs = []
ys = []
for user in os.listdir(aug_path):
    user = "crj"
    print(user)
    for condition in os.listdir(os.path.join(aug_path, user)):
        if "res_con" not in condition:
            continue
        print(condition)
        for imgs in tqdm(os.listdir(os.path.join(aug_path, user, condition))):
            imgs = "5476.png"
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
                continue
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
                
                    noise_point = np.random.multivariate_normal([np.mean(innerpoints[:, 0]), np.mean(innerpoints[:, 1])], [[np.std(innerpoints[:, 0]), 0], [0, np.std(innerpoints[:, 1])]], 10000)
                    for point in noise_point:
                        if int(point[0]) < 224 and int(point[1]) < 384:
                            aug_dis[int(point[0]), int(point[1])] += 1 / 10000 * 255

            sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            
            # aug_dis = gaussian_filter(binary, sigma=5)
            # aug_dis = (aug_dis - np.min(aug_dis)) / (np.max(aug_dis) - np.min(aug_dis)) * 255.0

            sal_img_resize = cv2.resize(sal_img, (16, 12), interpolation=cv2.INTER_LANCZOS4)
            aug_resize = cv2.resize(aug_dis, (16, 12), interpolation=cv2.INTER_LANCZOS4)

            # sal_img_resize = cv2.resize(sal_img, (76, 44), interpolation=cv2.INTER_LANCZOS4)
            # aug_resize = cv2.resize(aug_dis, (76, 44), interpolation=cv2.INTER_LANCZOS4)
            
            sal_flat = get_signature_from_heatmap(sal_img_resize)
            aug_flat = get_signature_from_heatmap(aug_resize)
            
            emd = 0
            emd, lowerbound, flow_matrix = cv2.EMD(sal_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)
            print(emd)

            aug_dis = aug_dis.astype(dtype=np.float32) * 255.0
            aug_resize = aug_resize.astype(dtype=np.float32) * 255.0
            # print(emd)
            if len(innerpoints) == 0 or np.isnan(emd):
                pass
            else:
                xs.append(emd)
                idx.append(imgs)
                ys.append(label)
                
            if label == 1:
                print(imgs)
                cv2.imwrite("saliency.png", sal_img)
                cv2.imwrite("saliency_resize.png", sal_img_resize)
                cv2.imwrite("aug.png", aug_dis)
                cv2.imwrite("aug_resize.png", aug_resize)
                exit()
            
        
        df = pd.DataFrame({"name": idx, "emd": xs, "label": ys})
        df.to_csv("./data_{}_{}.csv".format(condition, user))
        exit()