from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import sklearn.metrics as metrics

# fig, axes = plt.subplots(1, 1)

df_con = pd.read_csv("./data_con.csv")
df_gaze = pd.read_csv("./data_gaze.csv")
df = pd.concat([df_con, df_gaze])

class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
class_count_0, class_count_1 = df['label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

test = pd.concat([class_1_over, class_0], axis=0)

X = test['emd']
y = test['label']

min_emd = np.min(X)
max_emd = np.max(X)

emds = []
accs = []

for i in range(100):
    cur_emd = (max_emd - min_emd) * i / 100 + min_emd
    pred_y = []
    for x, y_true in zip(X, y):
        if x < cur_emd:
            pred_y.append(1)
        else:
            pred_y.append(0)
    emds.append(cur_emd)
    accs.append(metrics.accuracy_score(y, pred_y))

plt.plot(emds, accs)
plt.show()

