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
from matplotlib.ticker import PercentFormatter

data_path = "./metrics"

files = []
for user in os.listdir(data_path):
    if "DS_Store" in user:
        continue
    if "zyh" not in user:
        continue
    for condition in os.listdir(os.path.join(data_path, user)):
        if "cafe" not in condition:
            continue
        print(condition)
        files.append(pd.read_csv(os.path.join(data_path, user, condition)))
cafe_data = pd.concat(files)
files = []
for user in os.listdir(data_path):
    if "DS_Store" in user:
        continue
    if "zyh" not in user:
        continue
    for condition in os.listdir(os.path.join(data_path, user)):
        if "lab" not in condition:
            continue
        print(condition)
        files.append(pd.read_csv(os.path.join(data_path, user, condition)))
lab_data = pd.concat(files)
files = []
for user in os.listdir(data_path):
    if "DS_Store" in user:
        continue
    if "zyh" not in user:
        continue
    for condition in os.listdir(os.path.join(data_path, user)):
        if "home" not in condition:
            continue
        print(condition)
        files.append(pd.read_csv(os.path.join(data_path, user, condition)))
home_data = pd.concat(files)


def test(train_set, test_set):
    class_0 = train_set[train_set['label'] == 0]
    class_1 = train_set[train_set['label'] == 1]
    class_count_0, class_count_1 = train_set['label'].value_counts()
    class_1_over = class_1.sample(class_count_0, replace=True)

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(class_0['emd_g_a'], bins=100, weights=np.ones(len(class_0['emd_g_a'])) / len(class_0['emd_g_a']))
    axes[1].hist(class_1['emd_g_a'], color="#ff7f0e", bins=100, weights=np.ones(len(class_1['emd_g_a'])) / len(class_1['emd_g_a']))
    axes[0].set_xlabel("EMD")
    axes[0].set_ylabel("Percentage of frames")
    axes[1].set_xlabel("EMD")
    axes[1].set_ylabel("Percentage of frames")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

    test = pd.concat([class_1_over, class_0], axis=0)

    X = test['emd_g_a']
    y = test['label']

    X = X.values.reshape(-1, 1)
    clf = LogisticRegression()
    clf.fit(X, y)

    class_0 = test_set[test_set['label'] == 0]
    class_1 = test_set[test_set['label'] == 1]
    class_count_0, class_count_1 = test_set['label'].value_counts()
    class_1_over = class_1.sample(class_count_0, replace=True)

    test = pd.concat([class_1_over, class_0], axis=0)

    X_test = test['emd_g_a']
    y_test = test['label']

    X_test = X_test.values.reshape(-1, 1)

    print(clf.score(X_test, y_test))

test(home_data, pd.concat([lab_data, cafe_data]))
test(lab_data, pd.concat([home_data, cafe_data]))
test(cafe_data, pd.concat([lab_data, home_data]))
