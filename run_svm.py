from sklearn import svm
import pandas as pd
import os
import sklearn.metrics as metrics
from matplotlib import pyplot as plt

data_path = "./data"

for test_user in os.listdir(data_path):
    print(test_user)
    files = []
    test_files = []
    for user in os.listdir(data_path):
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

    X_train = test['emd_ani_sal']
    y_train = test['label']
    X_test = test_df['emd_ani_sal']
    y_test = test_df['label']
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)

    clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

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