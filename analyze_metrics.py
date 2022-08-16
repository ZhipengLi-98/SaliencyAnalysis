import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
import threading
from scipy.ndimage import gaussian_filter
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

metrics_path = "./metrics"

dfs = []

for user in os.listdir(metrics_path):
    user = "djx"
    for files in os.listdir(os.path.join(metrics_path, user)):
        dfs.append(pd.read_csv(os.path.join(metrics_path, user, files)))
    break
    
df = pd.concat(dfs)

class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
class_count_0, class_count_1 = df['label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

print(class_count_0, class_count_1)

test = pd.concat([class_1_over, class_0], axis=0)

features = ["frequency", "locationx", "locationy", "strength"]
X = test.loc[:, features].values
y = np.array(test["label"]).reshape(-1, 1)

print(X.shape)
print(y.shape)

# clf = svm.SVC()
clf = LogisticRegression()
# clf = RandomForestClassifier()
clf.fit(X, y)
print(clf.score(X, y))

# plt.scatter(df["frequency"], df["label"])
# plt.show()
