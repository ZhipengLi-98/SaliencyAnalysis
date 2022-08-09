import os

imgs_path = "./imgs"


for user in os.listdir(imgs_path):
    print(user)
    for condition in os.listdir(os.path.join(imgs_path, user)):
        print(condition)
        for folder in os.listdir(os.path.join(imgs_path, user, condition)):
            if folder.split("_")[-1].split(".")[0] == "all":
        break
    break