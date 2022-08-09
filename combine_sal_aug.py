import os
from tqdm import tqdm
import cv2
import numpy as np

import keras
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout
from keras import Sequential
from keras.applications.vgg16 import VGG16

aug_path = "./augs"
sal_path = "./saliency"

X = []
y = []

n_frames = 15

user = "zxyx"
for condition in os.listdir(os.path.join(aug_path, user)):
    if "gaze" in condition:
        continue
    images = sorted(os.listdir(os.path.join(aug_path, user, condition)))
    for i in range(n_frames, len(images)):
        past_imgs = images[i - n_frames: i]
        sample = []
        label = 0
        for imgs in past_imgs:
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
            sample.append(sal_img)
        X.append(sample)
        y.append(label)
        break

X = np.array(X)
X = X.reshape(-1, n_frames, 224, 384, 1)
y = np.array(y)

# create a VGG16 "model", we will use
# image with shape (224, 224, 3)
vgg = VGG16(
    include_top=False,
    weights=None,
    input_shape=(224, 384, 1)
)
# do not train first layers, I want to only train
# the 4 last layers (my own choice, up to you)
for layer in vgg.layers[:-4]:
    layer.trainable = False
# create a Sequential model
model = Sequential()
# add vgg model for 5 input images (keeping the right shape
model.add(
    TimeDistributed(vgg, input_shape=(n_frames, 224, 384, 1))
)
# now, flatten on each output to send 5 
# outputs with one dimension to LSTM
model.add(
    TimeDistributed(
        Flatten()
    )
)
model.add(LSTM(64, activation='relu', return_sequences=False))
# finalize with standard Dense, Dropout...
model.add(Dense(16, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1, activation='softmax'))
model.compile('adam', loss='categorical_crossentropy')

print(model.summary())
model.fit(X, y, epochs=2, batch_size=64, verbose=2)
# evaluate
result = model.predict(X, batch_size=64, verbose=0)
print(result)
