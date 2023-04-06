import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def get_time_frame(hour=0, min=0, sec=0, frame_rate=None):
    specific_frame = (hour * 3600 + min * 60 + sec) * frame_rate
    return int(specific_frame)

cap = cv.VideoCapture('driving.mp4')

frame_rate = cap.get(cv.CAP_PROP_FPS)    
    


frame_num = get_time_frame(hour=0, min=0, sec=10, frame_rate=frame_rate)

cap.set(1, frame_num)
ret, frame = cap.read()
