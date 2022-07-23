from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

fig, axes = plt.subplots(1, 1)

df = pd.read_csv("./data_gaze.csv")

class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
class_count_0, class_count_1 = df['label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

test = pd.concat([class_1_over, class_0], axis=0)

X = test['emd']
y = test['label']

axes.scatter(X, y)
# plt.show()

X = X.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = svm.SVC(probability=True)
scores = cross_val_score(clf, X, y, cv=5)
print(np.mean(scores))

clf = LogisticRegression(random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print(np.mean(scores))

clf = RandomForestClassifier(random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print(np.mean(scores))

# for i in df[df['label'] == 1]["emd"]:
#     print(clf.predict_proba(np.array([i]).reshape(-1, 1)))