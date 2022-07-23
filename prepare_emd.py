import os
import cv2
from matplotlib.cbook import simple_linear_interpolation 
import numpy as np
from tqdm import tqdm
from pyemd import emd, emd_samples
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance

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

aug_path = "./augs"
sal_path = "./saliency"

# idx = []
# xs = []
# ys = []

# for user in os.listdir(aug_path):
#     user = "crj"
#     print(user)
#     for condition in os.listdir(os.path.join(aug_path, user)):
#         if "con" not in condition:
#             continue
#         print(condition)
#         for imgs in tqdm(os.listdir(os.path.join(aug_path, user, condition))):
#             # print(imgs)
#             img = cv2.imread(os.path.join(aug_path, user, condition, imgs))
#             gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
#             contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             innerpoints = []
#             label = 0
#             outline = []
#             aug_dis = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
#             if len(contours) == 0:
#                 # no augmentation
#                 continue
#             else:
#                 outline = contours[0]
#                 if len(contours) > 1:
#                     temp = []
#                     num = 0
#                     for c in contours:
#                         if len(c) > num:
#                             temp = c
#                             num = len(c)
#                     outline = temp
#                 for i in range(img.shape[0]):
#                     for j in range(img.shape[1]):
#                         if cv2.pointPolygonTest(outline, (j, i), False) > 0:
#                             innerpoints.append([i, j, gray[i][j]])
                
#                 if len(innerpoints) > 0:
#                     innerpoints = np.array(innerpoints)
#                     if (np.mean(innerpoints[:, 2])) < 70:
#                         print(imgs, np.mean(np.mean(innerpoints[:, 2])))
#                         label = 1
                
#                     noise_point = np.random.multivariate_normal([np.mean(innerpoints[:, 0]), np.mean(innerpoints[:, 1])], [[np.std(innerpoints[:, 0]), 0], [0, np.std(innerpoints[:, 1])]], 1000)
#                     for point in noise_point:
#                         aug_dis[int(point[0]), int(point[1])] += 1 / 1000 * 255

#             sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            
#             sal_img_resize = cv2.resize(sal_img, (38, 22), interpolation=cv2.INTER_LANCZOS4)
#             # sal_img_resize /= np.sum(sal_img_resize)
#             aug_resize = cv2.resize(aug_dis, (38, 22), interpolation=cv2.INTER_LANCZOS4)
#             # aug_resize /= np.sum(aug_resize)
            
#             sal_flat = get_signature_from_heatmap(sal_img_resize)
#             aug_flat = get_signature_from_heatmap(aug_resize)
#             emd = 0
#             try:
#                 emd, lowerbound, flow_matrix = cv2.EMD(sal_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)
#             except:
#                 print(imgs)
#                 continue
            
#             if len(innerpoints) == 0 or np.isnan(emd):
#                 pass
#             else:
#                 xs.append(emd)
#                 idx.append(imgs)
#                 ys.append(label)
            
#         df = pd.DataFrame({"name": idx, "emd": xs, "label": ys})
#         df.to_csv("./data_con.csv")
#     break  
            
idx = []
xs = []
ys = []    
for user in os.listdir(aug_path):
    user = "crj"
    print(user)
    for condition in os.listdir(os.path.join(aug_path, user)):
        if "gaze" not in condition:
            continue
        print(condition)
        temp_x = []
        temp_idx = []
        temp_y = []
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
                
                    noise_point = np.random.multivariate_normal([np.mean(innerpoints[:, 0]), np.mean(innerpoints[:, 1])], [[np.std(innerpoints[:, 0]), 0], [0, np.std(innerpoints[:, 1])]], 1000)
                    for point in noise_point:
                        aug_dis[int(point[0]), int(point[1])] += 1 / 1000 * 255

            sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            
            sal_img_resize = cv2.resize(sal_img, (38, 22), interpolation=cv2.INTER_LANCZOS4)
            # sal_img_resize /= np.sum(sal_img_resize)
            aug_resize = cv2.resize(aug_dis, (38, 22), interpolation=cv2.INTER_LANCZOS4)
            # aug_resize /= np.sum(aug_resize)
            
            sal_flat = get_signature_from_heatmap(sal_img_resize)
            aug_flat = get_signature_from_heatmap(aug_resize)
            emd = 0
            try:
                emd, lowerbound, flow_matrix = cv2.EMD(sal_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)
            except:
                print(imgs)
                continue
            
            if len(innerpoints) == 0 or np.isnan(emd):
                pass
            else:
                if label == 0:
                    temp_x.append(emd)
                    temp_idx.append(imgs)
                    temp_y.append(label)
                elif label == 1:
                    xs.extend(temp_x[:-45] if len(xs) > 45 else [])
                    idx.extend(temp_idx[:-45] if len(xs) > 45 else [])
                    ys.extend(temp_y[:-45] if len(xs) > 45 else [])
                    temp_x = []
                    temp_y = []
                    temp_idx = []
                    xs.append(emd)
                    idx.append(imgs)
                    ys.append(label)
            
        df = pd.DataFrame({"name": idx, "emd": xs, "label": ys})
        df.to_csv("./data_gaze.csv")
        exit()
    break  
                
