import os
import cv2 as cv
from LPR import LPR
import numpy as np

########DETECTION ON IMAGE##########

lpr = LPR('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')
#input_image = lpr.read_image(r'C:\Users\Pc\Desktop\OCR\images\325.jpg')
input_image = lpr.read_image(r'plaka.jpg')
lpr.display_plt(input_image, 'original')

di = lpr.detect_image(input_image, 25, minSize=None)
for i in di:
    lpr.display_plt(i, 'Crops')

    
lpr.display_plt(di[1], 'Matches')


di[1] = np.mean(di[1], axis=2).astype(np.uint8)
lpr.display_plt(di[1], 'Matches')

###########SPLIT FOR OCR############

orig, inv = lpr.inverse_threshold(di[1], adaptive=True, isDisplay=True)

lpr.floodfill(inv, 0, 4)
lpr.display_plt(inv, 'filled')

lpr.floodfill(inv, 22, 15)
lpr.display_plt(inv, 'filled')

inv = lpr.crop(inv, upper_crop_size=5, lower_crop_size=5)
lpr.display_plt(inv, 'Horizontal Crop')

characters = lpr.split(orig, inv, minWhitePix=0, maxWhitePix=2, isWrite=False)

if input_image.shape[0] // di[0].shape[0] >= 15:
    for i in range(len(characters)):
        characters[i] = lpr.crop(characters[i], upper_crop_size=10, lower_crop_size=15)

m = 0
for i in characters:
    lpr.display_plt(i, 'Char')
    cv.imwrite(f'ROI/ROI_{m}.jpg', i)
    m += 1
####################################


##########OCR FOR IMAGE##################

os.chdir(r'C:\Users\Pc\Desktop\OCR\OCR')
from tensorflow.keras.models import load_model

model = load_model('ocr_80.h5')

predictions = ''
classes = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
       'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
       'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z']

for i in range(len(characters)):
    characters[i] = cv.resize(characters[i], (80, 200), interpolation = cv.INTER_AREA)
    predict_result = model.predict(characters[i].reshape(1, 200, 80))
    result = np.argmax(predict_result)
    predictions += classes[result]

print(predictions)
##########################################

