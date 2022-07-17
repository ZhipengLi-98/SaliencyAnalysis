import os
import cv2 
import numpy as np
from tqdm import tqdm
from pyemd import emd, emd_samples
import pandas as pd

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

aug_path = "./augs"
sal_path = "./saliency"

xs = []
ys = []

for user in os.listdir(aug_path):
    user = "crj"
    print(user)
    for condition in os.listdir(os.path.join(aug_path, user)):
        print(condition)
        for imgs in tqdm(os.listdir(os.path.join(aug_path, user, condition))):
            # print(imgs)
            img = cv2.imread(os.path.join(aug_path, user, condition, imgs))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = contours[0]
            if len(contours) > 1:
                temp = []
                num = 0
                for c in contours:
                    if len(c) > num:
                        temp = c
                        num = len(c)
                outline = temp
            innerpoints = []
            flag = True
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if cv2.pointPolygonTest(outline, (j, i), False) > 0:
                        innerpoints.append(gray[i][j])
                        if gray[i][j] > 250:
                            flag = False
                    if not flag:
                        break
                if not flag:
                    break
            
            label = 0

            if (np.mean(innerpoints)) < 70 and flag:
                print(imgs, np.mean(innerpoints))
                # sig_binary = img_to_sig2(binary)
                # sig_sal = img_to_sig2(sal_img)
                # print(emd_samples(sig_binary, sig_sal))
                label = 1
            sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE)
            xs.append(emd_samples(binary, sal_img))
            ys.append(label)
        df = pd.DataFrame({"emd": xs, "label": ys})
        df.to_csv("./data.csv")
        exit()
    break  
                
