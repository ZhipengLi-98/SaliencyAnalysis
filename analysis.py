import os
import numpy as np
import pandas as pd
import scipy.stats as stats

data_path = "./reaction_time"

color = []
scale = []
pos = []

for user in os.listdir(data_path):
    for condition in os.listdir(os.path.join(data_path, user)):
        df = pd.read_csv(os.path.join(data_path, user, condition))
        time = df["ReactionTime"].to_numpy()
        if "color" in condition:
            color.append(np.mean(time))
        elif "scale" in condition:
            scale.append(np.mean(time))
        elif "pos" in condition:
            pos.append(np.mean(time))

print(len(color), len(scale), len(pos))
print(np.mean(color), np.mean(scale), np.mean(pos))
fvalue, pvalue = stats.f_oneway(color, scale, pos)
print(fvalue, pvalue)

typing = []
video = []

for user in os.listdir(data_path):
    for condition in os.listdir(os.path.join(data_path, user)):
        df = pd.read_csv(os.path.join(data_path, user, condition))
        time = df["ReactionTime"].to_numpy()
        if "typing" in condition:
            typing.append(np.mean(time))
        elif "video" in condition:
            video.append(np.mean(time))

print(len(typing), len(video))
print(np.mean(typing), np.mean(video))
fvalue, pvalue = stats.f_oneway(typing, video)
print(fvalue, pvalue)

physicalhome1 = []
physicalhome2 = []
virtualhome = []
virtuallab = []

for user in os.listdir(data_path):
    for condition in os.listdir(os.path.join(data_path, user)):
        df = pd.read_csv(os.path.join(data_path, user, condition))
        time = df["ReactionTime"].to_numpy()
        if "physicalhome1" in condition:
            physicalhome1.append(np.mean(time))
        elif "physicalhome2" in condition:
            physicalhome2.append(np.mean(time))
        elif "virtualhome" in condition:
            virtualhome.append(np.mean(time))
        elif "virtuallab" in condition:
            virtuallab.append(np.mean(time))

print(len(physicalhome1), len(physicalhome2), len(virtualhome), len(virtuallab))
print(np.mean(physicalhome1), np.mean(physicalhome2), np.mean(virtualhome), np.mean(virtuallab))
fvalue, pvalue = stats.f_oneway(physicalhome1, physicalhome2, virtualhome, virtuallab)
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(physicalhome1, virtualhome)
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(physicalhome2, virtuallab)
print(fvalue, pvalue)

users = []
background = []
animation = []
task = []
reaction = []
for user in os.listdir(data_path):
    for condition in os.listdir(os.path.join(data_path, user)):
        df = pd.read_csv(os.path.join(data_path, user, condition))
        time = df["ReactionTime"].to_numpy()
        for t in time:
            users.append(user)
            background.append(condition.split(".")[0].split("_")[1])
            animation.append(condition.split(".")[0].split("_")[3])
            task.append(condition.split(".")[0].split("_")[2])
            reaction.append(t)

df = pd.DataFrame({"user": users, "background": background, "animation": animation, "task": task, "time": reaction})
df.to_csv("reactiontime.csv")
        