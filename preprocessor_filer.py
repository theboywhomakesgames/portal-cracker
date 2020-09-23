from pathlib import Path
import shutil
import numpy as np
import cv2
import glob

base_path = './ds_raw'
j = 0

def extract_digits(img, key, j):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 255, 100])

    mask = cv2.inRange(img_hsv, lower_black, upper_black)

    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erosion = cv2.erode(mask, kernel_e, iterations=1)
    dilate = cv2.dilate(erosion, kernel_d, iterations=3)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imgs = []
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        if w < 8 or h < 8:
            continue

        # Getting ROI
        roi = erosion[y:y+h, x:x+w]

        # show ROI
        try:
            digit = key[len(imgs)]
            imgs.append((roi, digit))

        except:
            continue

    if len(imgs) == 5:
        for digit_img, digit in imgs:
            j += 1
            cv2.imwrite('saved.png', digit_img)
            shutil.move('./saved.png', './ppds/' + digit + '/' + str(j) + '.png')

    return j


for foldername in glob.glob(base_path + '/*'):
    key = foldername.replace(base_path, '').replace('\\', '')

    for filename in glob.glob(foldername + '/*.png'):
        fname = filename.replace(foldername, '').replace('\\', '')
        img = cv2.imread(filename)
        j = extract_digits(img, key, j)
