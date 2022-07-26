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
import os
import torch
from torch import nn
import math

# fig, axes = plt.subplots(1, 1)

data_path = "./data"

files = []
for user in os.listdir(data_path):
    user = "zz"
    for condition in os.listdir(os.path.join(data_path, user)):
        files.append(pd.read_csv(os.path.join(data_path, user, condition)))
    break

df = pd.concat(files)

class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
class_count_0, class_count_1 = df['label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

test = pd.concat([class_1_over, class_0], axis=0)

const_emd = math.sqrt(16 * 16 + 12 * 12)

X = test['emd']
y = test['label']

print(len(X))

min_emd = np.min(X)
max_emd = np.max(X)

emds = []
accs = []

highest_roc_auc = 0
highest_fpr = 0
highest_tpr = 0

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

    fpr, tpr, threshold = metrics.roc_curve(y, pred_y)
    roc_auc = metrics.auc(fpr, tpr)
    if roc_auc > highest_roc_auc:
        highest_roc_auc = roc_auc
        highest_fpr = fpr
        highest_tpr = tpr

plt.plot(highest_fpr, highest_tpr, 'b', label = 'AUC = %0.2f' % highest_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

plt.plot(emds, accs)
plt.show()
