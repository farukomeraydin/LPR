import os
import cv2 as cv
from LPR import LPR
import numpy as np

from tensorflow.keras.models import load_model

model = load_model('ocr_80.h5')

classes = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
       'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
       'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z']

lpr = LPR('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')
input_image = lpr.read_image(r'images\501.jpg') #528 322 1002 501 511
lpr.display_plt(input_image, 'original')


di = lpr.detect_image(input_image, 25)
for i in di:
    lpr.display_plt(i, 'Crops')



lpr.tryout(di, 0, 2, 1, 6, lower_crop_size=5, isDisplay=True)
lpr.tryout(di, 2, 2, 1, 6, lower_crop_size=5, isDisplay=True)