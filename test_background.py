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
    if "tmh" not in user:
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
    if "tmh" not in user:
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
    if "tmh" not in user:
        continue
    for condition in os.listdir(os.path.join(data_path, user)):
        if "home" not in condition:
            continue
        print(condition)
        files.append(pd.read_csv(os.path.join(data_path, user, condition)))
home_data = pd.concat(files)

def test(train_set, test):
    class_0 = train_set[train_set['label'] == 0]
    class_1 = train_set[train_set['label'] == 1]
    class_count_0, class_count_1 = train_set['label'].value_counts()
    class_1_over = class_1.sample(class_count_0, replace=True)

    test = pd.concat([class_1_over, class_0], axis=0)

    X = test['emd_gaze_aug']
    y = test['label']

    X = X.values.reshape(-1, 1)
    clf = LogisticRegression()
    print(clf.fit(X, y))

    for test_set in test:

        class_0 = test_set[test_set['label'] == 0]
        class_1 = test_set[test_set['label'] == 1]
        class_count_0, class_count_1 = test_set['label'].value_counts()
        class_1_over = class_1.sample(class_count_0, replace=True)

        test = pd.concat([class_1_over, class_0], axis=0)

        X_test = test['emd_gaze_aug']
        y_test = test['label']

        X_test = X_test.values.reshape(-1, 1)

        print(clf.score(X_test, y_test))

test(home_data, [lab_data, cafe_data])
test(lab_data, [home_data, cafe_data])
test(cafe_data, [lab_data, home_data])
exit()
df = pd.concat(files)

class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
class_count_0, class_count_1 = df['label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

test = pd.concat([class_1_over, class_0], axis=0)

X = test['emd_gaze_aug']
y = test['label']

# print(len(X))

# fig, axes = plt.subplots(1, 2)
# axes[0].hist(class_0['emd'], bins=100, weights=np.ones(len(class_0['emd'])) / len(class_0['emd']))
# axes[1].hist(class_1['emd'], color="#ff7f0e", bins=100, weights=np.ones(len(class_1['emd'])) / len(class_1['emd']))
# axes[0].set_xlabel("EMD")
# axes[0].set_ylabel("Percentage of frames")
# axes[1].set_xlabel("EMD")
# axes[1].set_ylabel("Percentage of frames")
# axes[0].yaxis.set_major_formatter(PercentFormatter(1))
# axes[1].yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()
# exit()

X = X.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = svm.SVC(probability=True)
# scores = cross_val_score(clf, X, y, cv=5)
# print(np.mean(scores))

clf = LogisticRegression()
clf.fit(X_train, y_train)
# scores = cross_val_score(clf, X, y, cv=5)
# print(np.mean(scores))

probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
print(threshold)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# for user in os.listdir(data_path):
#     print(user)
#     files = []
#     for condition in os.listdir(os.path.join(data_path, user)):
#         files.append(pd.read_csv(os.path.join(data_path, user, condition)))

#     df = pd.concat(files)

#     class_0 = df[df['label'] == 0]
#     class_1 = df[df['label'] == 1]
#     class_count_0, class_count_1 = df['label'].value_counts()
#     class_1_over = class_1.sample(class_count_0, replace=True)

#     test = pd.concat([class_1_over, class_0], axis=0)

#     X = df['emd_gaze_aug']
#     y = df['label']

#     print(len(X))

#     # fig, axes = plt.subplots(1, 1)
#     # axes.hist(class_0['emd'])
#     # axes.hist(class_1_over['emd'])
#     # plt.show()
#     # exit()

#     X = X.values.reshape(-1, 1)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     # clf = svm.SVC(probability=True)
#     # scores = cross_val_score(clf, X, y, cv=5)
#     # print(np.mean(scores))

#     clf = LogisticRegression()
#     clf.fit(X_train, y_train)
#     # scores = cross_val_score(clf, X, y, cv=5)
#     # print(np.mean(scores))

#     probs = clf.predict_proba(X_test)
#     preds = probs[:,1]
#     fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
#     print(threshold)
#     roc_auc = metrics.auc(fpr, tpr)

#     plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#     plt.legend(loc = 'lower right')
#     plt.plot([0, 1], [0, 1],'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()


for test_user in os.listdir(data_path):
    print(test_user)
    files = []
    test_files = []
    for user in os.listdir(data_path):
        print(user)
        if user == test_user:
            for condition in os.listdir(os.path.join(data_path, test_user)):
                test_files.append(pd.read_csv(os.path.join(data_path, test_user, condition)))
        else:
            for condition in os.listdir(os.path.join(data_path, user)):
                files.append(pd.read_csv(os.path.join(data_path, user, condition)))
        
    df = pd.concat(files)
    test_df = pd.concat(test_files)

    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]
    class_count_0, class_count_1 = df['label'].value_counts()
    class_1_over = class_1.sample(class_count_0, replace=True)

    test = pd.concat([class_1_over, class_0], axis=0)

    X = test_df['emd_gaze_aug']
    y = test_df['label']

    print(len(X))

    X = X.values.reshape(-1, 1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X
    y_train = y
    X_test = test_df['emd_gaze_aug']
    y_test = test_df['label']
    X_test = X_test.values.reshape(-1, 1)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # clf = svm.SVC(probability=True)
    # scores = cross_val_score(clf, X, y, cv=5)
    # print(np.mean(scores))

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    # scores = cross_val_score(clf, X, y, cv=5)
    # print(np.mean(scores))

    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    print(threshold)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()