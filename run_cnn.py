import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from random import random
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import keras
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout, MaxPool2D
from keras import Sequential
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

imgs_path = "./formal/imgs"
saliency_path = "./formal/saliency"
data_path = "./data"

Xs = {}
ys = {}

for user in os.listdir(saliency_path):
    temp_x = []
    temp_label = []
    temp_y = []
    for condition in os.listdir(os.path.join(saliency_path, user)):
        condition = condition.split(".csv")[0]
        condition = condition.split("_all.mp4")[0]
        print(user, condition)
        aug_path = os.path.join(imgs_path, user, condition, condition + "_ani.mp4")
        gaze_path = os.path.join(imgs_path, user, condition, condition + "_gaze.mp4")
        sal_path = os.path.join(saliency_path, user, condition + "_all.mp4")
        df = pd.read_csv("./data/{}/{}.csv".format(user, condition.split("_all.mp4")[0]))
        # df = pd.read_csv("./data/{}/data_{}_{}_{}.csv".format(user, condition.split("_")[1], condition.split("_")[2], user))
        idx = df["index"].tolist()
        X = df["emd_ani_sal"].tolist()
        # X = df["emd_ani_gaze"].tolist()
        y = df["label"].tolist()
        
        if len(X) == 0:
            print(user, condition)
            continue

        for i in tqdm(range(len(idx))):
            imgs = "frame" + str(idx[i]) + ".jpg"
            label = y[i]
            img = cv2.imread(os.path.join(aug_path, imgs))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            sal_img = cv2.imread(os.path.join(sal_path, "%04d.png" % (idx[i] + 1)), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            gaze_img = cv2.imread(os.path.join(gaze_path, imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)
            ret, binary_gaze = cv2.threshold(gaze_img, 20, 255, cv2.THRESH_BINARY)
            gaze_dis = gaussian_filter(binary_gaze, sigma=5)
            gaze_dis = (gaze_dis - np.min(gaze_dis)) / (np.max(gaze_dis) - np.min(gaze_dis)) * 255.0
            gaze_dis = gaze_dis.astype(sal_img.dtype)

            aug_dis = gaussian_filter(binary, sigma=5)
            aug_dis = (aug_dis - np.min(aug_dis)) / (np.max(aug_dis) - np.min(aug_dis)) * 255.0
            aug_dis = aug_dis.astype(sal_img.dtype)

            merge = cv2.merge((aug_dis, sal_img))
            merge = merge / 255.0
            merge = merge.reshape(224, 384, 2)
            merge_resize = cv2.resize(merge, (40, 24), interpolation=cv2.INTER_LANCZOS4)

            sal_img = np.maximum(sal_img, aug_dis)
            sal_img = sal_img / 255.0
            sal_img = sal_img.reshape(224, 384, 1)
            sal_img_resize = cv2.resize(sal_img, (40, 24), interpolation=cv2.INTER_LANCZOS4)

            # cv2.imshow("test", merge)
            # cv2.imwrite("overlap.png", sal_img_resize)
            # cv2.waitKey()
            # exit()

            temp_x.append(merge_resize)
            cur_y = np.zeros(2)
            cur_y[label] = 1
            temp_label.append(cur_y)
            temp_y.append(label)
    
    df = pd.DataFrame({"img": temp_x, "label": temp_label, "temp_label": temp_y})
    class_0 = df[df['temp_label'] == 0]
    class_1 = df[df['temp_label'] == 1]
    class_count_0, class_count_1 = df['temp_label'].value_counts()
    class_1_over = class_1.sample(class_count_0, replace=True)

    test = pd.concat([class_1_over, class_0], axis=0)

    X = test['img'].tolist()
    y = test['label'].tolist()

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(-1, 24, 40, 1)
    y = y.reshape(-1, 2)

    Xs[user] = X
    ys[user] = y

    if len(Xs.keys()) > 5:
        break

for test_user in os.listdir(saliency_path):
    print(test_user)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for user in os.listdir(saliency_path):
        if user != test_user and user in Xs.keys():
            X_train.extend(Xs[user])
            y_train.extend(ys[user])

    # train_test_length = int(len(Xs[test_user]) / 10)
    # X_train.extend(Xs[test_user][:train_test_length])
    # y_train.extend(ys[test_user][:train_test_length])
    # X_test.extend(Xs[test_user][train_test_length:])
    # y_test.extend(ys[test_user][train_test_length:])

    X_test = Xs[test_user]
    y_test = ys[test_user]

    # for i in range(len(Xs[test_user])):
    #     if random() > 0.9:
    #         X_train.append(Xs[test_user][i])
    #         y_train.append(ys[test_user][i])
    #     else:
    #         X_test.append(Xs[test_user][i])
    #         y_test.append(ys[test_user][i])

    X_train = np.array(X_train).reshape(-1, 24, 40, 2)
    y_train = np.array(y_train).reshape(-1, 2)
    X_test = np.array(X_test).reshape(-1, 24, 40, 2)
    y_test = np.array(y_test).reshape(-1, 2)

    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=5, activation="relu", input_shape=(24, 40, 2)))
    model.add(Conv2D(32, kernel_size=5, activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    #train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, verbose=2)
    print("test loss, test acc:", results)

    y_pred = model.predict(X_test).ravel()
    y_test = y_test.flatten()
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    print(test_user, roc_auc)

    # import visualkeras
    # from PIL import ImageFont

    # font = ImageFont.truetype("arial.ttf", 15)  # using comic sans is strictly prohibited!
    # visualkeras.layered_view(model, to_file='output.png', legend=True, font=font).show() # write and show

# model.save("cnn.h5")