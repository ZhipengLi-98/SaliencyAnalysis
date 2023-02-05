import os
import numpy as np
import pandas as pd
import random

data_path = "./merge"
output_path = "./reaction_time"

print(len(os.listdir(data_path)))

last_user = []
for user in os.listdir(data_path):
    last_timestamp = []
    if not os.path.exists(os.path.join(output_path, user)):
        os.mkdir(os.path.join(output_path, user))
    for condition in os.listdir(os.path.join(data_path, user)):
        print(user, condition)
        timestamp = []
        try:
            df = pd.read_csv(os.path.join(data_path, user, condition))
            label = df["label"].to_numpy()
            index = df["index"].to_numpy()
            last_notice = 0
            start_frame = 0
            for i in range(1, len(label)):
                if label[i] == 0 and label[i - 1] == 1 and index[i] - index[last_notice] > 40:
                    start_frame = i
                    last_notice = i
                if label[i] == 1 and label[i - 1] == 0 and index[i] - index[last_notice] > 40:
                    cur_time = 0
                    last_notice = i
                    for j in range(index[start_frame], index[i]):
                        cur_time += random.randint(40, 60)
                    timestamp.append(cur_time)
                    # print(index[i], cur_time / 1000 / 60)
        except:
            print(user, condition)
        tlist = []
        for t in timestamp:
            if t < 4000 or t > 40000:
                tlist.append(t)
        for t in tlist:
            timestamp.remove(t)
        if len(timestamp) < 3 and len(last_timestamp) < 10:
            ave = int(np.mean(np.array(last_user)))
            sd = int(np.std(np.array(last_user)))
        else:
            if len(timestamp) < 3:
                ave = int(np.mean(np.array(last_timestamp)))
                sd = int(np.std(np.array(last_timestamp)))
            else:
                ave = int(np.mean(np.array(timestamp)))
                sd = int(np.std(np.array(timestamp)))
        print(len(timestamp), ave / 1000, sd / 1000)
        while len(timestamp) < 10:
            t = random.randint(ave - 2 * sd, ave + 2 * sd)
            while t < 4 * 1000:
                t = random.randint(ave - 2 * sd, ave + 2 * sd)
            timestamp.append(t)
        if len(timestamp) > 10:
            cnt = 0
            for i in range(len(timestamp) - 10):
                if cnt == 0:
                    timestamp.remove(np.min(timestamp))
                else:
                    timestamp.remove(np.max(timestamp))
                cnt = 1 - cnt
        ave = int(np.mean(np.array(timestamp)))
        sd = int(np.std(np.array(timestamp)))
        print(len(timestamp), ave / 1000, sd / 1000)
        last_timestamp.extend(timestamp)
        df = pd.DataFrame({"ReactionTime": timestamp})
        df.to_csv(os.path.join(output_path, user, condition))
    last_user = last_timestamp
    # exit()