import os

import numpy as np
import cv2
from keras.callbacks import TensorBoard, ModelCheckpoint
np.random.seed(42)
import scipy.misc as mc
data_location = ''
training_images_loc = data_location + 'RC_SLO/train/image/'
training_label_loc = data_location + 'RC_SLO/train/label/'
testing_images_loc = data_location + 'RC_SLO/test/image/'
testing_label_loc = data_location + 'RC_SLO/test/label/'
train_files = os.listdir(training_images_loc)
train_data = []
train_label = []

desired_size=368
for i in train_files:
    im = mc.imread(training_images_loc + i)
    label = mc.imread(training_label_loc + i.split(".")[0] + "_GT.tif")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label,(desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)
train_data = np.array(train_data)

train_label = np.array(train_label)

test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []



x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)

from  RCAUNet import *
model=RCAUNet(input_size=(desired_size,desired_size,3))
weight="Model/RC_SLO/RCAUNet.h5"

restore=False
if restore and os.path.isfile(weight):
    model.load_weights(weight)

model_checkpoint = ModelCheckpoint(weight, monitor='val_acc', verbose=1, save_best_only=True)

model.fit(x_train, y_train,
                epochs=300,
                batch_size=1,
                validation_split=0.1,
                # validation_data=(x_test, y_test),
                shuffle=True,

                callbacks= [TensorBoard(log_dir='./autoencoder'), model_checkpoint])

