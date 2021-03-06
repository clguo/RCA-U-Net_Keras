import os
import cv2
import numpy as np
from sklearn.metrics import  recall_score, roc_auc_score, accuracy_score, confusion_matrix
from keras.callbacks import  ModelCheckpoint
import scipy.misc as mc

data_location = ''
testing_images_loc = data_location + 'Luna/test/image/'
testing_label_loc = data_location + 'Luna/test/label/'


test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []

for i in test_files:
    test_data.append(cv2.resize((mc.imread(testing_images_loc + i)), (512, 512)))
    # Change '_manual1.tiff' to the label name
    temp = cv2.resize(mc.imread(testing_label_loc + i.split('.')[0] + '_mask.tif'),
                      (512, 512))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)
test_data = np.array(test_data)
test_label = np.array(test_label)


x_test = test_data.astype('float32') / 255.

y_test = test_label.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), 512, 512, 3))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), 512, 512, 1))  # adapt this if using `channels_first` im


from  RCAUNet import *
model=RCAUNet(input_size=(512,512,3))
weight="Model/Luna/RCAUNet.h5"

if os.path.isfile(weight): model.load_weights(weight)

model_checkpoint = ModelCheckpoint(weight, monitor='val_acc', verbose=1, save_best_only=True)

y_pred = model.predict(x_test)
y_pred_threshold = []
i=0
for y in y_pred:

    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    y = y * 255
    cv2.imwrite('./Luna/test/result/%d.png' % i, y)
    i+=1
y_test = list(np.ravel(y_test))
y_pred_threshold = list(np.ravel(y_pred_threshold))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

print('Accuracy:', accuracy_score(y_test, y_pred_threshold))
print('Sensitivity:', recall_score(y_test, y_pred_threshold))
print('Specificity', tn / (tn + fp))
print('NPV', tn / (tn + fn))
print('PPV', tp / (tp + fp))
print('AUC:', roc_auc_score(y_test, list(np.ravel(y_pred))))
print("F1:",2*tp/(2*tp+fn+fp))