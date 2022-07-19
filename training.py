from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 1)

df = pd.read_csv("./data.csv")

X = df['emd']
y = df['label']

# axes[0].scatter(X, y)

df = pd.read_csv("./data_test.csv")

class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]
class_count_0, class_count_1 = df['label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

test = pd.concat([class_1_over, class_0], axis=0)

X = test['emd']
y = test['label']

axes.scatter(X, y)
axes.set_xlim(0, 0.2)
plt.show()

exit()

X = X.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = svm.SVC(probability=True).fit(X_train, y_train)
print(clf.score(X_test, y_test))

# for i in df[df['label'] == 0]["emd"]:
#     print(clf.predict_proba(np.array([i]).reshape(-1, 1)))