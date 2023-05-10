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
    ret, screenshot = cap.read()
    #screenshot = wincap.get_screenshot()
    processed_image = vision_lp.apply_hsv_filter(screenshot)  
    edges_image = vision_lp.apply_edge_filter(processed_image)
    #rectangles = vision_lp.find(processed_image, 0.46)
    #output_image = vision_lp.draw_rectangles(screenshot, rectangles)
    
    keypoint_image = edges_image
    
    x, w, y, h = [200, 1130, 70, 750]
    keypoint_image = keypoint_image[y:y+h, x:x+w]
    kp1, kp2, matches, match_points = vision_lp.match_keypoints(keypoint_image)
    match_image = cv.drawMatches(vision_lp.needle_img, kp1, keypoint_image, kp2, matches, None)
    
    if match_points:
        # find the center point of all the matched features
        center_point = vision_lp.centeroid(match_points)
        # account for the width of the needle image that appears on the left
        center_point[0] += vision_lp.needle_w
        # drawn the found center point on the output image
        match_image = vision_lp.draw_crosshairs(match_image, [center_point])

    
    
    cv.imshow('Keypoint Search', match_image)
    cv.imshow('Processed', processed_image)
    cv.imshow('Edges', edges_image)
    #cv.imshow('Matches', output_image)
    
    print(f'FPS {1 / (time() - loop_time)}')
    loop_time = time()
    
    if cv.waitKey(1) == ord('q'):
        break

    

cap.release()
cv.destroyAllWindows()

print('Done.')