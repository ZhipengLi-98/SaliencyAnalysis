from genericpath import exists
import cv2
import os
from tqdm import tqdm

imgs_path = "./imgs"
aug_path = "./augs"

for user in os.listdir(imgs_path):
    print(user)
    for condition in os.listdir(os.path.join(imgs_path, user)):
        print(condition)
        for folder in os.listdir(os.path.join(imgs_path, user, condition)):
            if folder.split("_")[-1].split(".")[0] == "aug":
                input_path = os.path.join(imgs_path, user, condition, folder)
                output_path = os.path.join(aug_path, user, folder)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                for i in tqdm(range(len(os.listdir(input_path)))):
                    img = cv2.imread(os.path.join(input_path) + "/frame%d.jpg" % i)
                    dsize = (384, 224)
                    output = cv2.resize(img, dsize)
                    cv2.imwrite(os.path.join(output_path) + "/%04d.png" % (i + 1), output)

