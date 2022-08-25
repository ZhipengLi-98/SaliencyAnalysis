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
    idx = []
    xs = []
    emd_g_a = []
    ys = []
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
                
                    # noise_point = np.random.multivariate_normal([np.mean(innerpoints[:, 0]), np.mean(innerpoints[:, 1])], [[np.std(innerpoints[:, 0]), 0], [0, np.std(innerpoints[:, 1])]], 1000)
                    # for point in noise_point:
                    #     if int(point[0]) < 224 and int(point[1]) < 384:
                    #         aug_dis[int(point[0]), int(point[1])] += 1 / 1000 * 255

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

            # sal_img_resize = cv2.resize(sal_img, (76, 44), interpolation=cv2.INTER_LANCZOS4)
            # aug_resize = cv2.resize(aug_dis, (76, 44), interpolation=cv2.INTER_LANCZOS4)
            
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
                xs.append(emd)
                emd_g_a.append(emd_gaze_aug)
                idx.append(imgs)
                ys.append(label)
        except:
            print(user, condition, imgs)
                
    df = pd.DataFrame({"name": idx, "emd": xs, "emd_gaze_aug": emd_g_a, "label": ys})
    df.to_csv("./data_{}_{}_{}.csv".format(condition.split("_")[1], condition.split("_")[2], user))

def run_gaze(user, condition, images):
    idx = []
    xs = []
    emd_g_a = []
    ys = []    
    temp_x = []
    temp_idx = []
    temp_emd_g_a = []
    temp_y = []
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
                
                    # noise_point = np.random.multivariate_normal([np.mean(innerpoints[:, 0]), np.mean(innerpoints[:, 1])], [[np.std(innerpoints[:, 0]), 0], [0, np.std(innerpoints[:, 1])]], 1000)
                    # for point in noise_point:
                    #     if int(point[0]) < 224 and int(point[1]) < 384:
                    #         aug_dis[int(point[0]), int(point[1])] += 1 / 1000 * 255

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

            # sal_img_resize = cv2.resize(sal_img, (76, 44), interpolation=cv2.INTER_LANCZOS4)
            # aug_resize = cv2.resize(aug_dis, (76, 44), interpolation=cv2.INTER_LANCZOS4)
            
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
                    temp_x.append(emd)
                    temp_emd_g_a.append(emd_gaze_aug)
                    temp_idx.append(imgs)
                    temp_y.append(label)
                elif label == 1:
                    xs.extend(temp_x[:-45] if len(temp_x) > 45 else [])
                    emd_g_a.extend(temp_emd_g_a[:-45] if len(temp_emd_g_a) > 45 else [])
                    idx.extend(temp_idx[:-45] if len(temp_idx) > 45 else [])
                    ys.extend(temp_y[:-45] if len(temp_y) > 45 else [])
                    temp_x = []
                    temp_emd_g_a = []
                    temp_y = []
                    temp_idx = []
                    xs.append(emd)
                    emd_g_a.append(emd_gaze_aug)
                    idx.append(imgs)
                    ys.append(label)
        except:
            print(user, condition, imgs)
                
    df = pd.DataFrame({"name": idx, "emd": xs, "emd_gaze_aug": emd_g_a, "label": ys})
    df.to_csv("./data_{}_{}_{}.csv".format(condition.split("_")[1], condition.split("_")[2], user))

if __name__ == "__main__":
    for user in os.listdir(aug_path):
        user = "cyr"
        for condition in os.listdir(os.path.join(aug_path, user)):
            print(user, condition)
            images = sorted(os.listdir(os.path.join(aug_path, user, condition)))
            if "gaze" in condition:
                run_gaze(user, condition, images)
            else:
                run_con(user, condition, images)
        break

# class myThread (threading.Thread):
#     def __init__(self, user, condition, images):
#         threading.Thread.__init__(self)
#         self.user = user
#         self.name = condition
#         self.images = images
#     def run(self):
#         print("Starting " + self.name)
#         if "gaze" in self.name:
#             run_gaze(self.user, self.name, self.images)
#         else:
#             run_con(self.user, self.name, self.images)
#         print("Exiting " + self.name)

# temp = []
# for user in os.listdir(aug_path):
#     user = "gzt"
#     for condition in os.listdir(os.path.join(aug_path, user)):
#         images = sorted(os.listdir(os.path.join(aug_path, user, condition)))
#         temp.append(myThread(user, condition, images))
#     break
# for t in temp:
#     t.start()
# for t in temp:
#     t.join()
