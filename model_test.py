from pathlib import Path
import shutil
import numpy as np
import cv2
import glob

import tensorflow as tf
from tensorflow import keras

base_path = './ds_raw'
j = 0

model = keras.models.load_model('./dg_recog_model')
model.summary()


def extract_digits(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 255, 100])

    mask = cv2.inRange(img_hsv, lower_black, upper_black)

    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erosion = cv2.erode(mask, kernel_e, iterations=1)
    dilate = cv2.dilate(erosion, kernel_d, iterations=3)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        if w < 8 or h < 8:
            continue

        # Getting ROI
        roi = erosion[y:y+h, x:x+w]

        # show ROI
        cv2.imwrite('saved.png', roi)
        roi = cv2.imread('./saved.png')
        roi = cv2.resize(roi, (28, 28))
        roi = tf.expand_dims(roi, 0)
        prediction = model.predict(roi)
        print(str(np.argmax(prediction) + 1))

    cv2.imshow("img", img)
    cv2.waitKey(0)


img = cv2.imread('./1.png')
extract_digits(img)
