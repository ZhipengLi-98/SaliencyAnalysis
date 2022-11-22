import pandas as pd
import os

data_path = "./data"

for user in os.listdir(data_path):
    user = "gww"
    for f in os.listdir(os.path.join(data_path, user)):
        print(f)
        df = pd.read_csv(os.path.join(data_path, user, f))
        idx = df["index"].to_list()
        emd_ass = df["emd_ani_sal"].to_list()
        emd_ags = df["emd_ani_gaze"].to_list()
        labels = df["label"].to_list()
        for i in range(len(idx)):
            if labels[i] == 1:
                if (idx[i] + 30) in idx:
                    print(i, idx[i])
                    # labels[i] = 0
        # break
    break
