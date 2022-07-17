from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

df = pd.read_csv("./data.csv")

X = df['emd']
y = df['label']

plt.scatter(X, y)
plt.show()

print(df[df['label'] == 1])

X = X.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = svm.SVC().fit(X_train, y_train)
print(clf.score(X_test, y_test))