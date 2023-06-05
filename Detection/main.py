import cv2 as cv
import numpy as np
from time import time
from windowcapture import WindowCapture
from vision import Vision
import matplotlib.pyplot as plt

def get_time_frame(hour=0, min=0, sec=0, frame_rate=None):
    specific_frame = (hour * 3600 + min * 60 + sec) * frame_rate
    return int(specific_frame)

"""
wincap = WindowCapture()
cap = cv.VideoCapture('driving2_1080p.mkv')

frame_rate = cap.get(cv.CAP_PROP_FPS)    
    
frame_num = get_time_frame(hour=0, min=3, sec=10, frame_rate=frame_rate)

cap.set(1, frame_num)
"""
cascade_lp = cv.CascadeClassifier('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')
cap = cv.VideoCapture('driving4.mp4')
vision_lp = Vision(None)
loop_time = time()
from time import sleep
i = 1

#cv.namedWindow("Matches", cv.WINDOW_NORMAL)
#cv.resizeWindow("Matches", 480, 270)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(1, 1))
ret, frame = cap.read()

interval = frame.shape[1] // 36
dead_zone = int(interval * 7.5)
k = 0
while True:
    ret, screenshot = cap.read()
    screenshot = screenshot[:, dead_zone:screenshot.shape[1] - dead_zone, :]
    rectangles = cascade_lp.detectMultiScale(screenshot, minNeighbors=3, minSize=(24, 72), maxSize=(24, 72))
    if len(rectangles):
        for i in range(len(rectangles)):
            x = rectangles[i][0]
            y = rectangles[i][1]
            w = rectangles[i][2]
            h = rectangles[i][3]
            crop = screenshot[y:y + h, x:x + w]
            gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
            th1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            
            #eroded_image = cv.erode(im_bw, kernel, iterations=1)
            #dilated_image = cv.dilate(im_bw, kernel, iterations=1)
            opening = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
            closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
            
            plt.figure()
            plt.imshow(closing, cmap='gray')
            plt.show()
            cv.imwrite(f'sample/{k}.jpg', closing)
            cv.imwrite(f'sample/gray_{k}.jpg', gray)
            k += 1
            
    detection_image = vision_lp.draw_rectangles(screenshot, rectangles)
    
    cv.imshow('Matches', detection_image)

    print(f'FPS {1 / (time() - loop_time + 0.000001)}')
    loop_time = time()
    
    key = cv.waitKey(1)
    #sleep(0.01)
    
    if key == ord('q'):
        break
    #elif key == ord('f'):
    #    cv.imwrite(f'positive2/p{loop_time}.jpg', screenshot)
    #elif key == ord('d'):
    #    cv.imwrite(f'negative/{loop_time}.jpg', screenshot)
    
    

    

cap.release()
cv.destroyAllWindows()

print('Done.')
