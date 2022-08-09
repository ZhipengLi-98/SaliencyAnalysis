import os
from re import U
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import keras
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout
from keras import Sequential
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

aug_path = "./augs"
sal_path = "./saliency"

X = []
y = []
temp_y = []

for user in os.listdir(aug_path):
    for condition in os.listdir(os.path.join(aug_path, user)):
        print(user, condition)
        images = sorted(os.listdir(os.path.join(aug_path, user, condition)))
        df = pd.read_csv("./data/{}/data_{}_{}_{}.csv".format(user, condition.split("_")[1], condition.split("_")[2], user))
        for imgs in tqdm(images):
            try:
                label = int(df[df["name"] == imgs]["label"])
                img = cv2.imread(os.path.join(aug_path, user, condition, imgs))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)

                aug_dis = gaussian_filter(binary, sigma=5)
                aug_dis = (aug_dis - np.min(aug_dis)) / (np.max(aug_dis) - np.min(aug_dis)) * 255.0

                sal_img = np.maximum(sal_img, aug_dis)
                sal_img = sal_img / 255.0
                sal_img = sal_img.reshape(224, 384, 1)
                sal_img_resize = cv2.resize(sal_img, (16, 12), interpolation=cv2.INTER_LANCZOS4)
                X.append(sal_img_resize)
                cur_y = np.zeros(2)
                cur_y[label] = 1
                y.append(cur_y)
                temp_y.append(label)
            except:
                pass

df = pd.DataFrame({"img": X, "label": y, "temp_label": temp_y})
class_0 = df[df['temp_label'] == 0]
class_1 = df[df['temp_label'] == 1]
class_count_0, class_count_1 = df['temp_label'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

test = pd.concat([class_1_over, class_0], axis=0)

X = test['img'].tolist()
y = test['label'].tolist()

X = np.array(X)
y = np.array(y)

X = X.reshape(-1, 12, 16, 1)
y = y.reshape(-1, 2)

model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(12, 16, 1)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

model.save("cnn.h5")