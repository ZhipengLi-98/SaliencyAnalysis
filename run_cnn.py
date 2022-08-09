import os
from tqdm import tqdm
import cv2
import numpy as np

import keras
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout
from keras import Sequential
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

aug_path = "./augs"
sal_path = "./saliency"

X = []
y = []

user = "zxyx"
for condition in os.listdir(os.path.join(aug_path, user)):
    if "gaze" in condition:
        continue
    images = sorted(os.listdir(os.path.join(aug_path, user, condition)))
    for imgs in images:
        img = cv2.imread(os.path.join(aug_path, user, condition, imgs))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        innerpoints = []
        label = 0
        outline = []
        aug_dis = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
        if len(contours) == 0:
            # no augmentation
            continue
        else:
            outline = contours[0]
            if len(contours) > 1:
                temp = []
                num = 0
                for c in contours:
                    if len(c) > num:
                        temp = c
                        num = len(c)
                outline = temp
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if cv2.pointPolygonTest(outline, (j, i), False) > 0:
                        innerpoints.append([i, j, gray[i][j]])
            
            if len(innerpoints) > 0:
                innerpoints = np.array(innerpoints)
                if (np.mean(innerpoints[:, 2])) < 70:
                    print(imgs, np.mean(np.mean(innerpoints[:, 2])))
                    label = 1
    
        sal_img = cv2.imread(os.path.join(sal_path, user, condition.split("aug")[0] + "all.mp4", imgs), cv2.IMREAD_GRAYSCALE).astype(dtype=np.float32)

        for i in range(sal_img.shape[0]):
            for j in range(sal_img.shape[1]):
                sal_img[i, j] = max(sal_img[i, j], gray[i, j])
        sal_img = sal_img / 255.0
        sal_img = sal_img.reshape(224, 384, 1)
        sal_img_resize = cv2.resize(sal_img, (16, 12), interpolation=cv2.INTER_LANCZOS4)
        X.append(sal_img_resize)
        y.append(label)
        break

X = np.array(X)
y = np.array(y)

print(X.shape)

model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(12, 16, 1)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(1, activation="softmax"))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)