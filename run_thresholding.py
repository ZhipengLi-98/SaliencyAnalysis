import pandas as pd
import os
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import random

data_path = "./merge"

fig = plt.figure(figsize=(12, 6))

test_user_num = 1

test_user_list = [[i] for i in os.listdir(data_path)]
# test_user_list = [random.sample(os.listdir(data_path), test_user_num) for i in range(100)]

aucs = []
for test_user in test_user_list:
    print(test_user)
    files = []
    test_files = []
    for user in os.listdir(data_path):
        if user in test_user:
            for condition in os.listdir(os.path.join(data_path, user)):
                test_files.append(pd.read_csv(os.path.join(data_path, user, condition)))
        else:
            for condition in os.listdir(os.path.join(data_path, user)):
                files.append(pd.read_csv(os.path.join(data_path, user, condition)))

    df = pd.concat(files)
    test_df = pd.concat(test_files)

    emd_ass = df["emd_ani_sal"]
    labels = df["label"]

    min_emd_ass = emd_ass.min()
    max_emd_ass = emd_ass.max()

    emd_ass = emd_ass.to_list()
    labels = labels.to_list()

    y_pred = []
    for j in tqdm(range(len(emd_ass))):
        y_pred.append(1 - emd_ass[j] / (max_emd_ass - min_emd_ass))
    fpr, tpr, threshold = metrics.roc_curve(labels, y_pred)

    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(threshold[index], ndigits = 4)
    gmeanOpt = round(gmean[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)
    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    print((1 - thresholdOpt) * (max_emd_ass - min_emd_ass))
    roc_auc = metrics.auc(fpr, tpr)
    
    aucs.append(roc_auc)
    print(roc_auc)

    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right', fontsize="small", bbox_to_anchor=(1.2, 0))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.suptitle("Average AUC = %0.2f" % np.mean(np.array(aucs)), fontsize=16)
    fig.tight_layout()
    # plt.show()
    plt.savefig("./thresholding_leave_{}_user_out_temp.jpg".format(test_user_num))