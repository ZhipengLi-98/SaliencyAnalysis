from genericpath import exists
import cv2
import os
from tqdm import tqdm

imgs_path = "./imgs"
gaze_path = "./gaze"

for user in os.listdir(imgs_path):
    user = "cyr"
    print(user)
    for condition in os.listdir(os.path.join(imgs_path, user)):
        print(condition)
        for folder in os.listdir(os.path.join(imgs_path, user, condition)):
            if folder.split("_")[-1].split(".")[0] == "gaze":
                input_path = os.path.join(imgs_path, user, condition, folder)
                output_path = os.path.join(gaze_path, user, folder)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                for i in tqdm(range(len(os.listdir(input_path)))):
                    img = cv2.imread(os.path.join(input_path) + "/frame%d.jpg" % i)
                    dsize = (384, 224)
                    output = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(os.path.join(output_path) + "/%04d.png" % (i + 1), output)
    break

