import LPR
import os
import cv2 as cv

os.chdir(r'C:\Users\faruk\OneDrive\Masaüstü\haar')

########DETECTION ON IMAGE##########

lpr = LPR.LPR('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')
input_image = lpr.read_image('home.jpg')

di = lpr.detect_image(input_image, 25)
lpr.display_img(di, 'Matches')
####################################


########DETECTION ON VIDEO##########

lpr = LPR.LPR('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')
kernel = cv.getStructuringElement(cv.MORPH_RECT,(1, 1))
lpr.detect_video('driving4.mp4', kernel, maxSize=(24, 72), isWrite=False, printFPS=True, isDisplay=False)

####################################


###########SPLIT FOR OCR############

lpr = LPR.LPR(None)

img = lpr.read_image('hd.jpg', isGray=True)
lpr.display_img(img, 'CROPPED')

orig, inv = lpr.inverse_threshold(img)

lpr.floodfill(inv, 0, 5)
lpr.display_plt(inv, 'filled')

lpr.floodfill(inv, 23, 30)
lpr.display_plt(inv, 'filled')

characters = lpr.split(orig, inv)

for i in characters:
    lpr.display_plt(i, 'Char')
####################################