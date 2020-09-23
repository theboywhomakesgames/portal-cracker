import glob
import os
import cv2
import numpy as np
from pathlib import Path
import shutil

path = './ds_raw'
cur_index = 0
cur_path = ''

for filename in glob.glob(path + '/*.png'):
    keys = filename.replace(path, '').replace('\\', '').split('__')
    index = keys[0]
    file_index = keys[1]

    if cur_index != index:
        cur_index = index

        img = cv2.imread(filename)
        cv2.imshow("preview", img)
        cv2.waitKey(1)
        name = input()
        cur_path = './' + name

        Path(cur_path).mkdir(parents=True, exist_ok=True)

        cv2.destroyWindow("preview")

    shutil.move(filename, cur_path + '/' + file_index)
