import numpy as np
import cv2

img = cv2.imread('./data/1 (5).png')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_black = np.array([0, 0, 0])
upper_black = np.array([255, 255, 100])

mask = cv2.inRange(img_hsv, lower_black, upper_black)

kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
erosion = cv2.erode(mask, kernel_e, iterations=1)
dilate = cv2.dilate(erosion, kernel_d, iterations=3)
cv2.imshow("eroded", erosion)
cv2.imshow("dilated", dilate)

contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    if w < 10 or h < 10:
        continue

    # Getting ROI
    roi = erosion[y:y+h, x:x+w]

    # show ROI
    cv2.imshow("roi" + str(i), roi)

cv2.waitKey(0)
