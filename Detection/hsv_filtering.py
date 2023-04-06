import cv2 as cv
import numpy as np
from time import time
from windowcapture import WindowCapture
from vision import Vision
from hsvfilter import HsvFilter

#WindowCapture.list_window_names()


wincap = WindowCapture()
cap = cv.VideoCapture('driving.mp4')
vision_lp = Vision('hsv1.jpg')
vision_lp.init_control_gui()

hsv_filter = HsvFilter(0, 0, 115, 179, 255, 255, 0, 0, 20, 0)
loop_time = time()
_, start_frame = cap.read()
i = 1
while True:
    #screenshot = wincap.get_screenshot()
    ret, screenshot = cap.read()
    """
    difference = cv.absdiff(screenshot, start_frame)
    threshold = cv.threshold(difference, 25, 255, cv.THRESH_BINARY)[1]
    start_frame = screenshot
    if threshold.sum() > 1:
        print(threshold.sum())
        processed_image = vision_lp.apply_hsv_filter(screenshot, hsv_filter)
    #screenshot = cv.resize(screenshot, (960, 540))
    #cv.imshow('Computer Vision', screenshot)
    
        rectangles = vision_lp.find(processed_image, 0.35)
        output_image = vision_lp.draw_rectangles(screenshot, rectangles)

    else:
        output_image = screenshot
    """
    
    processed_image = vision_lp.apply_hsv_filter(screenshot, hsv_filter)   
    cv.imshow('Processed', processed_image)
    #cv.imshow('Matches', output_image)
    
    print(f'FPS {1 / (time() - loop_time)}')
    loop_time = time()
    
    if cv.waitKey(1) == ord('q'):
        break
    
    if cv.waitKey(1) == ord('s'):
        cv.imwrite(f'hsv_images/{i}.jpg', processed_image)
        i += 1
    

cap.release()
cv.destroyAllWindows()

print('Done.')
