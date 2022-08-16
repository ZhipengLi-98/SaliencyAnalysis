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
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

metrics_path = "./metrics"

dfs = {}

for user in os.listdir(metrics_path):
    temp = []
    for files in os.listdir(os.path.join(metrics_path, user)):
        temp.append(pd.read_csv(os.path.join(metrics_path, user, files)))
    dfs[user] = temp

train = []
for user in os.listdir(metrics_path):
    train.extend(dfs[user])
df_train = pd.concat(train)
class_0 = df_train[df_train['label'] == 0]
class_1 = df_train[df_train['label'] == 1]
class_count_0, class_count_1 = df_train['label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

df_train = pd.concat([class_1_over, class_0], axis=0)
features = ["frequency", "locationx", "locationy", "strength", "emd_s_a", "emd_g_a"]

X = df_train.loc[:, features].values
y = np.array(df_train["label"]).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LogisticRegression()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


for test_user in os.listdir(metrics_path):
    train = []
    test = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for user in os.listdir(metrics_path):
        if user != test_user:
            train.extend(dfs[user])
        else:
            test.extend(dfs[user])
    
    df_train = pd.concat(train)
    df_test = pd.concat(test)

    class_0 = df_train[df_train['label'] == 0]
    class_1 = df_train[df_train['label'] == 1]
    class_count_0, class_count_1 = df_train['label'].value_counts()
    class_1_over = class_1.sample(class_count_0, replace=True)

    df_train = pd.concat([class_1_over, class_0], axis=0)

    features = ["frequency", "locationx", "locationy", "strength", "emd_s_a", "emd_g_a"]

    X_train = df_train.loc[:, features].values
    y_train = np.array(df_train["label"]).reshape(-1, 1)

    X_test = df_test.loc[:, features].values
    y_test = np.array(df_test["label"]).reshape(-1, 1)

    # train_test_length = int(len(X_test) / 10)
    # print(train_test_length)
    # print(X_test.shape)
    # print(X_test[:train_test_length])
    # X_train = np.concatenate((X_train, X_test[:train_test_length]))
    # y_train = np.concatenate((y_train, y_test[:train_test_length]))
    # X_test = X_test[train_test_length:]
    # y_test = y_test[train_test_length:]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
exit()


df = pd.concat(dfs)

class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
class_count_0, class_count_1 = df['label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

print(class_count_0, class_count_1)

test = pd.concat([class_1_over, class_0], axis=0)

features = ["frequency", "locationx", "locationy", "strength", "emd_s_a", "emd_g_a"]
features = ["frequency", "locationx", "locationy", "strength", "emd_s_a", "emd_g_a"]
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
