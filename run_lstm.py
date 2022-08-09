import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D

aug_path = "./augs"

n_frames = 60

Xs = {}
ys = {}

X_train = []
y_train = []

for user in os.listdir(aug_path):
    t_x = []
    t_y = []
    for condition in os.listdir(os.path.join(aug_path, user)):
        print(user, condition)
        df = pd.read_csv("./data/{}/data_{}_{}_{}.csv".format(user, condition.split("_")[1], condition.split("_")[2], user))
        X = df["emd"].to_numpy()
        y = df["label"].to_numpy()
        temp_x = []
        temp_y = []
        for i in range(n_frames, len(X)):
            temp_x.append(X[i - n_frames: i])
            temp_y.append(y[i])
        t_x.extend(temp_x)
        t_y.extend(temp_y)

    df = pd.DataFrame({"img": t_x, "label": t_y})
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]
    class_count_0, class_count_1 = df['label'].value_counts()
    class_1_over = class_1.sample(class_count_0, replace=True)

    test = pd.concat([class_1_over, class_0], axis=0)

    X = test['img'].tolist()
    y = test['label'].tolist()

    X = np.array(X)
    y = np.array(y)

    Xs[user] = X
    ys[user] = y

for test_user in os.listdir(aug_path):
    print(test_user)
    X_train = []
    y_train = []
    for user in os.listdir(aug_path):
        if user != test_user:
            X_train.extend(Xs[user])
            y_train.extend(ys[user])
    X_test = np.array(Xs[test_user])
    y_test = np.array(ys[test_user])

    X_train = np.array(X_train).reshape(-1, n_frames, 1)
    y_train = np.array(y_train).reshape(-1, 1, 1)

    print(X_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))
    
    # import visualkeras
    # from PIL import ImageFont
    # font = ImageFont.truetype("arial.ttf", 8)  # using comic sans is strictly prohibited!
    # visualkeras.layered_view(model, to_file='lstm.png', legend=True, font=font).show() # write and show
    # break