import os
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from random import random

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D

data_path = "./data"

n_frames = 10
data_per_condition = 1200

Xs = {}
ys = {}

for user in os.listdir(data_path):
    t_x = []
    t_y = []
    temp = []
    for condition in os.listdir(os.path.join(data_path, user)):
        # print(user, condition)
        temp_x = []
        temp_y = []
        df = pd.read_csv(os.path.join(data_path, user, condition))
        # df = pd.read_csv("./data/{}/data_{}_{}_{}.csv".format(user, condition.split("_")[1], condition.split("_")[2], user))
        idx = df["index"].tolist()
        X = df["emd_ani_sal"].tolist()
        # X = df["emd_ani_gaze"].tolist()
        y = df["label"].tolist()

        if len(X) == 0:
            print(user, condition)
            continue

        for i in range(n_frames, len(idx)):
            if idx[i] - idx[i - n_frames] == n_frames:
                temp_x.append(X[i - n_frames: i])
                temp_y.append(y[i])
        df = pd.DataFrame({"img": temp_x, "label": temp_y})
        class_0 = df[df['label'] == 0]
        class_1 = df[df['label'] == 1]

        class_0_resample = class_0.sample(data_per_condition, replace=True)
        class_1_resample = class_1.sample(data_per_condition, replace=True)
        temp.append(class_0_resample)
        temp.append(class_1_resample)

    # df = pd.DataFrame({"img": t_x, "label": t_y})
    # class_0 = df[df['label'] == 0]
    # class_1 = df[df['label'] == 1]
    # class_count_0, class_count_1 = df['label'].value_counts()
    # class_1_over = class_1.sample(class_count_0, replace=True)
    # # class_1_over = class_1.sample(10 * class_count_1, replace=True)
    # # class_0_down = class_0.sample(10 * class_count_1, replace=True)

    # test = pd.concat([class_1_over, class_0], axis=0)
    # # test = pd.concat([class_1_over, class_0_down], axis=0)
    test = pd.concat(temp, axis=0)

    X = test['img'].tolist()
    y = test['label'].tolist()

    X = np.array(X)
    y = np.array(y)

    Xs[user] = X
    ys[user] = y

for test_user in os.listdir(data_path):
    print(test_user)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for user in os.listdir(data_path):
        if user != test_user:
            X_train.extend(Xs[user])
            y_train.extend(ys[user])
    
    # train_test_length = int(len(Xs[test_user]) / 10)
    # X_train.extend(Xs[test_user][:train_test_length])
    # y_train.extend(ys[test_user][:train_test_length])
    # X_test.extend(Xs[test_user][train_test_length:])
    # y_test.extend(ys[test_user][train_test_length:])

    for i in range(len(Xs[test_user])):
        if random() > 1:
            X_train.append(Xs[test_user][i])
            y_train.append(ys[test_user][i])
        else:
            X_test.append(Xs[test_user][i])
            y_test.append(ys[test_user][i])
    
    X_train = np.array(X_train).reshape(-1, n_frames, 1)
    y_train = np.array(y_train).reshape(-1, 1, 1)
    X_test = np.array(X_test).reshape(-1, n_frames, 1)
    y_test = np.array(y_test).reshape(-1, 1, 1)

    print(X_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=150, batch_size=64, validation_data=(X_test, y_test))
    
    y_pred = model.predict(X_test).ravel()
    y_test = y_test.flatten()
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label = '{} AUC = %0.2f'.format(test_user) % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
plt.savefig("./lstm_results/leave_one_user_out.jpg")
    # plt.show()
    # break
    # import visualkeras
    # from PIL import ImageFont
    # font = ImageFont.truetype("arial.ttf", 8)  # using comic sans is strictly prohibited!
    # visualkeras.layered_view(model, to_file='lstm.png', legend=True, font=font).show() # write and show
    