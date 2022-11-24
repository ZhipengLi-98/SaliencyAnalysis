import pandas as pd
import os

data_path = "./data"
output_path = "./smooth"

for user in os.listdir(data_path):
    for f in os.listdir(os.path.join(data_path, user)):
        print(f)
        df = pd.read_csv(os.path.join(data_path, user, f))
        idx = df["index"].to_list()
        emd_ass = df["emd_ani_sal"].to_list()
        emd_ags = df["emd_ani_gaze"].to_list()
        labels = df["label"].to_list()
        for i in range(1, len(idx)):
            if idx[i] - idx[i - 2] == 2 and emd_ags[i - 2] != 0:
                if emd_ags[i] / emd_ags[i - 2] < 0.4:
                    labels[i] = 1
        for i in range(1, len(idx)):
            if labels[i - 1] == 1 and labels[i] == 0:
                if idx[i] + 90 not in idx:
                    labels[i] = 1
        for i in range(len(idx)):
            if labels[i] == 1:
                if (idx[i] + 60) in idx:
                    labels[i] = 0
        temp_idx = []
        temp_emd_ass = []
        temp_emd_ags = []
        temp_labels = []
        for i in range(len(idx)):
            if labels[i] == 1 and labels[i - 1] == 1:
                continue
            elif labels[i] == 0 and labels[i - 1] == 1:
                continue
            else:
                temp_idx.append(idx[i])
                temp_emd_ass.append(emd_ass[i])
                temp_emd_ags.append(emd_ags[i])
                temp_labels.append(labels[i])
        new_df = pd.DataFrame({"index": temp_idx, "emd_ani_sal": temp_emd_ass, "emd_ani_gaze": temp_emd_ags, "label": temp_labels})
        if not os.path.exists(os.path.join(output_path, user)):
            os.makedirs(os.path.join(output_path, user))
        new_df.to_csv(os.path.join(output_path, user, f))
    # break