import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from run_saliency import main
from prepare_emd import get_signature_from_heatmap

imgs_path = "./imgs"

for user in os.listdir(imgs_path):
    user = "zxyx"
    print(user)
    for condition in os.listdir(os.path.join(imgs_path, user)):
        print(condition)
        for folder in os.listdir(os.path.join(imgs_path, user, condition)):
            if folder.split("_")[-1].split(".")[0] == "all":
                images = sorted(os.listdir(os.path.join(imgs_path, user, condition, folder)))
                i = 0
                for img_name in images:
                    print(img_name)
                    img = cv2.imread(os.path.join(imgs_path, user, condition, folder, img_name))
                    print(img.shape)
                    white = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
                    image = cv2.circle(img, (420, 310), 20 + i, (255, 255, 255), -1)
                    cv2.imwrite("./generate/1/{}_{}".format(i, img_name), image)
                    image = cv2.circle(white, (420, 310), 20 + i, (255, 255, 255), -1)
                    aug_dis = gaussian_filter(image, sigma=5)
                    aug_dis = (aug_dis - np.min(aug_dis)) / (np.max(aug_dis) - np.min(aug_dis)) * 255.0
                    cv2.imwrite("./gen_aug/1/{}_{}".format(i, img_name), image)
                    i += 1
                    if i > 80:
                        break
        break
    break

input_path = "./generate/1/"
output_path = "./gen_saliency/1/"

main(input_path, output_path)

aug_path = "./gen_aug/1/"
for imgs in os.listdir(aug_path):
    aug_dis = cv2.imread(os.path.join(aug_path, imgs))
    aug_dis = cv2.cvtColor(aug_dis, cv2.COLOR_RGB2GRAY)
    sal_img = cv2.imread(os.path.join(output_path, "%04d.png" % (int(imgs.split("_")[0]) + 1)), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
    sal_img_resize = cv2.resize(sal_img, (40, 24), interpolation=cv2.INTER_LANCZOS4)
    aug_resize = cv2.resize(aug_dis, (40, 24), interpolation=cv2.INTER_LANCZOS4)
    
    sal_flat = get_signature_from_heatmap(sal_img_resize)
    aug_flat = get_signature_from_heatmap(aug_resize)
    emd, lowerbound, flow_matrix = cv2.EMD(sal_flat, aug_flat, distType=cv2.DIST_L2, lowerBound=0)

    print(imgs, emd)