import cv2
import os
from tqdm import tqdm

video_path = "./video/"
output_path = "./imgs/"

user = "djx"

user_path = os.path.join(video_path, user)
output_user_path = os.path.join(output_path, user)

if not os.path.exists(output_user_path):
    os.makedirs(output_user_path)

for folder in os.listdir(user_path):
    if not os.path.exists(os.path.join(output_user_path, folder)):
        os.makedirs(os.path.join(output_user_path, folder))
    for video in os.listdir(os.path.join(user_path, folder)):
        print(video)
        if not os.path.exists(os.path.join(output_user_path, folder, video)):
            os.makedirs(os.path.join(output_user_path, folder, video))
        vidcap = cv2.VideoCapture(os.path.join(user_path, folder, video))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(length)):
            success, image = vidcap.read()
            image = image[:, 80:]
            cv2.imwrite(os.path.join(output_user_path, folder, video) + "/frame%d.jpg" % i, image)   
