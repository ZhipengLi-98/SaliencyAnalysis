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

gaze_path = "./gaze"
aug_path = "./augs"
sal_path = "./saliency"

def run_con(user, condition, images):
    names = []
    locationx = []
    locationy = []
    strength = []
    frequency = []
    labels = []
    emd_s_a = []
    emd_g_a = []
    prev_img = []
    prev_cnt = 0
    for imgs in tqdm(images):
        try:
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
            if len(innerpoints) == 0:
                continue
            if len(prev_img) == 0:
                prev_img = binary
                prev_cnt = cv2.countNonZero(binary)
                continue
            
            sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            
            aug_dis = gaussian_filter(binary, sigma=5)
            aug_dis = (aug_dis - np.min(aug_dis)) / (np.max(aug_dis) - np.min(aug_dis)) * 255.0
            
            gaze_img = cv2.imread(os.path.join(gaze_path, user, condition.split("aug")[0] + "gaze.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            ret, binary_gaze = cv2.threshold(gaze_img, 20, 255, cv2.THRESH_BINARY)
            gaze_dis = gaussian_filter(binary_gaze, sigma=5)
            gaze_dis = (gaze_dis - np.min(gaze_dis)) / (np.max(gaze_dis) - np.min(gaze_dis)) * 255.0

            sal_img_resize = cv2.resize(sal_img, (16, 12), interpolation=cv2.INTER_LANCZOS4)
            aug_resize = cv2.resize(aug_dis, (16, 12), interpolation=cv2.INTER_LANCZOS4)
            gaze_resize = cv2.resize(gaze_dis, (16, 12), interpolation=cv2.INTER_LANCZOS4)
            sal_flat = get_signature_from_heatmap(sal_img_resize)
            aug_flat = get_signature_from_heatmap(aug_resize)
            gaze_flat = get_signature_from_heatmap(gaze_resize)
            emd = 0
            emd_gaze_aug = 0
            try:
                emd, lowerbound, flow_matrix = cv2.EMD(sal_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)
                emd_gaze_aug, lowerbound, flow_matrix = cv2.EMD(gaze_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)
            except:
                print(imgs)
                continue
            
            diff = cv2.countNonZero(cv2.absdiff(prev_img, binary))
            frequency.append(diff / (384 * 224))
            strength.append(diff / prev_cnt if prev_cnt > 0 else 0)
            names.append(imgs)
            locationx.append(np.mean(innerpoints[:, 0]) / 224)
            locationy.append(np.mean(innerpoints[:, 1]) / 384)
            labels.append(label)
            emd_g_a.append(emd_gaze_aug)
            emd_s_a.append(emd_s_a)
            prev_cnt = diff
            prev_img = binary
        except:
            print(imgs)
    df = pd.DataFrame({"name": names, "emd_g_a": emd_g_a, "emd_s_a": emd_s_a, "label": labels, "strength": strength, "frequency": frequency, "locationx": locationx, "locationy": locationy})
    df.to_csv("./metrics/{}/{}.csv".format(user, condition))


def run_gaze(user, condition, images):
    names = []
    locationx = []
    locationy = []
    strength = []
    frequency = []
    labels = []
    emd_s_a = []
    emd_g_a = []

    temp_names = []
    temp_lx = []
    temp_ly = []
    temp_s = []
    temp_f = []
    temp_l = []
    temp_s_a = []
    temp_g_a = []

    prev_img = []
    prev_cnt = 0
    for imgs in tqdm(images):
        # print(imgs)
        try:
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
                
            sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            
            aug_dis = gaussian_filter(binary, sigma=5)
            aug_dis = (aug_dis - np.min(aug_dis)) / (np.max(aug_dis) - np.min(aug_dis)) * 255.0

            gaze_img = cv2.imread(os.path.join(gaze_path, user, condition.split("aug")[0] + "gaze.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            ret, binary_gaze = cv2.threshold(gaze_img, 20, 255, cv2.THRESH_BINARY)
            gaze_dis = gaussian_filter(binary_gaze, sigma=5)
            gaze_dis = (gaze_dis - np.min(gaze_dis)) / (np.max(gaze_dis) - np.min(gaze_dis)) * 255.0

            sal_img_resize = cv2.resize(sal_img, (16, 12), interpolation=cv2.INTER_LANCZOS4)
            aug_resize = cv2.resize(aug_dis, (16, 12), interpolation=cv2.INTER_LANCZOS4)
            gaze_resize = cv2.resize(gaze_dis, (16, 12), interpolation=cv2.INTER_LANCZOS4)

            
            sal_flat = get_signature_from_heatmap(sal_img_resize)
            aug_flat = get_signature_from_heatmap(aug_resize)
            gaze_flat = get_signature_from_heatmap(gaze_resize)
            emd = 0
            emd_gaze_aug = 0
            try:
                emd, lowerbound, flow_matrix = cv2.EMD(sal_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)
                emd_gaze_aug, lowerbound, flow_matrix = cv2.EMD(gaze_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)
            except:
                print(imgs)
                continue
            
            if len(innerpoints) == 0 or np.isnan(emd):
                pass
            else:
                # Bug: do not consider the last one arrays
                if label == 0:
                    
                    diff = cv2.countNonZero(cv2.absdiff(prev_img, binary))
                    temp_f.append(diff / (384 * 224))
                    temp_s.append(diff / prev_cnt if prev_cnt > 0 else 0)
                    temp_names.append(imgs)
                    temp_lx.append(np.mean(innerpoints[:, 0]) / 224)
                    temp_ly.append(np.mean(innerpoints[:, 1]) / 384)
                    temp_l.append(label)
                    temp_g_a.append(emd_gaze_aug)
                    temp_s_a.append(emd_s_a)
                elif label == 1:
                    frequency.extend(temp_f[:-45] if len(temp_f) > 45 else [])
                    strength.extend(temp_s[:-45] if len(temp_s) > 45 else [])
                    names.extend(temp_names[:-45] if len(temp_names) > 45 else [])
                    locationx.extend(temp_lx[:-45] if len(temp_lx) > 45 else [])
                    locationy.extend(temp_ly[:-45] if len(temp_ly) > 45 else [])
                    labels.extend(temp_l[:-45] if len(temp_l) > 45 else [])
                    emd_g_a.extend(temp_g_a[:-45] if len(temp_g_a) > 45 else [])
                    emd_s_a.extend(temp_s_a[:-45] if len(temp_s_a) > 45 else [])
                    
                    temp_names = []
                    temp_lx = []
                    temp_ly = []
                    temp_s = []
                    temp_f = []
                    temp_l = []
                    temp_s_a = []
                    temp_g_a = []
            prev_cnt = diff
            prev_img = binary
        except:
            print(user, condition, imgs)
    df = pd.DataFrame({"name": names, "emd_g_a": emd_g_a, "emd_s_a": emd_s_a, "label": labels, "strength": strength, "frequency": frequency, "locationx": locationx, "locationy": locationy})
    df.to_csv("./metrics/{}/{}.csv".format(user, condition))

for user in os.listdir(aug_path):
    user = "zxyx"
    for condition in os.listdir(os.path.join(aug_path, user)):
        print(condition)
        images = sorted(os.listdir(os.path.join(aug_path, user, condition)))
        if "gaze" in condition:
            run_gaze(user, condition, images)
        else:
            run_con(user, condition, images)
    break