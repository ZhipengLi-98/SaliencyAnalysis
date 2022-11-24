import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np

data_path = "./new_data"

for user in os.listdir(data_path):
    user = "gww"
    print(user)
    files = []
    for condition in os.listdir(os.path.join(data_path, user)):
        condition = "gww_virtuallab_typing_color.csv"
        print(condition)
        files = []
        files.append(pd.read_csv(os.path.join(data_path, user, condition)))
        df = pd.concat(files)
        idx = df["index"].to_list()
        emd_ass = df["emd_ani_sal"].to_list()
        emd_ags = df["emd_ani_gaze"].to_list()
        labels = df["label"].to_list()
        fig, axes = plt.subplots(1, 1)
        axes.plot(idx, emd_ass, label="saliency")
        axes.plot(idx, emd_ags, label="gaze")
        axes.scatter(idx, labels)
        plt.legend()
        plt.show()
        break
    break
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(class_0['emd_ani_sal'], bins=100, weights=np.ones(len(class_0['emd_ani_sal'])) / len(class_0['emd_ani_sal']))
    axes[1].hist(class_1['emd_ani_sal'], color="#ff7f0e", bins=100, weights=np.ones(len(class_1['emd_ani_sal'])) / len(class_1['emd_ani_sal']))
    plt.show()
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(class_0['emd_ani_gaze'], bins=100, weights=np.ones(len(class_0['emd_ani_gaze'])) / len(class_0['emd_ani_gaze']))
    axes[1].hist(class_1['emd_ani_gaze'], color="#ff7f0e", bins=100, weights=np.ones(len(class_1['emd_ani_gaze'])) / len(class_1['emd_ani_gaze']))
    # plt.scatter(emd_ass, labels)
    # plt.scatter(emd_ags, labels)
    plt.show()
    break
