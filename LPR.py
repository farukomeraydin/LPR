import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from windowcapture import WindowCapture
from vision import Vision
from time import time, sleep

os.chdir(r'C:\Users\faruk\OneDrive\Masaüstü\haar')


class LPR:
    dim1 = 24
    dim2 = 72
    
    def __init__(self, cascade_path, vision=None):
        super().__init__()
        self.cascade = cv.CascadeClassifier('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')
        self.vision = Vision(vision)
        self.rectangles = None
        
    
    def read_image(self, img_path, isGray=False):
        if isGray is True:
            img = cv.imread(img_path, cv.COLOR_BGR2GRAY)
        else:
            img = cv.imread(img_path)
        if len(img.shape) > 2 and isGray is True:
            return np.mean(img, axis=2).astype(np.uint8)
        
        return img
    
    def detect_image(self, input_img, minNeighbors=3, minSize=(dim1, dim2), maxSize=None):
        interval = input_img.shape[1] // 36
        dead_zone = int(interval * 7.5)
        if len(input_img.shape) > 2:
            input_img = input_img[:, dead_zone:input_img.shape[1] - dead_zone, :]
        else:
            input_img = input_img[:, dead_zone:input_img.shape[1] - dead_zone]
            
        self.rectangles = self.cascade.detectMultiScale(input_img, minNeighbors=minNeighbors, minSize=minSize, maxSize=maxSize)
        detection_image = self.vision.draw_rectangles(input_img, self.rectangles)
        
        return detection_image
    
    def get_time_frame(hour=0, min=0, sec=0, frame_rate=None):
        specific_frame = (hour * 3600 + min * 60 + sec) * frame_rate
        return int(specific_frame)
    
    def detect_video(self, video_path, kernel, minNeighbors=3, minSize=(dim1, dim2), maxSize=None, isDisplay=True, printFPS=True, isWrite=True, writePath=None):
        cap = cv.VideoCapture(video_path)
        ret, frame = cap.read()
        interval = frame.shape[1] // 36
        dead_zone = int(interval * 7.5)
        if isWrite:
            k = 0
            
        if printFPS:
            loop_time = time()
        
        while True:
            ret, screenshot = cap.read()
            screenshot = screenshot[:, dead_zone:screenshot.shape[1] - dead_zone, :]
            self.rectangles = self.cascade.detectMultiScale(screenshot, minNeighbors=minNeighbors, minSize=minSize, maxSize=maxSize)
            if len(self.rectangles):
                for i in range(len(self.rectangles)):
                    x = self.rectangles[i][0]
                    y = self.rectangles[i][1]
                    w = self.rectangles[i][2]
                    h = self.rectangles[i][3]
                    crop = screenshot[y:y + h, x:x + w]
                    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                    th1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
                    th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
                    
                    #eroded_image = cv.erode(im_bw, kernel, iterations=1)
                    #dilated_image = cv.dilate(im_bw, kernel, iterations=1)
                    opening = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
                    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
                    if isDisplay:
                        plt.figure()
                        plt.imshow(closing, cmap='gray')
                        plt.show()
                        
                    if isWrite:
                        cv.imwrite(f'{writePath}/{k}.jpg', closing)
                        cv.imwrite(f'sample/gray_{k}.jpg', gray)
                        k += 1
                        
            detection_image = self.vision.draw_rectangles(screenshot, self.rectangles)
            
            cv.imshow('Matches', detection_image)
            
            if printFPS:
                print(f'FPS {1 / (time() - loop_time + 0.000001)}')
                loop_time = time()
            
            key = cv.waitKey(1)
            #sleep(0.01)
            
            if key == ord('q'):
                break
            
        cap.release()
        cv.destroyAllWindows()
    
        
    def display_plt(self, img, title):
        plt.figure(figsize=(12, 10))
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.show()
        
    def inverse_threshold(self, image, resize_dim=(dim2, dim1), th=100):
        inv = cv.bitwise_not(image)
        
        LPR.display_plt(self, inv, 'INV')

        inv_down = cv.resize(inv, resize_dim, cv.INTER_AREA)
        
        LPR.display_plt(self, inv_down, 'INV Downscaled')
        
        inv_down[inv_down > th] = 255
        inv_down[inv_down < th] = 0
        
        LPR.display_plt(self, inv_down, 'INV Thresholded')
        
        return image, inv_down
    
        
    def floodfill(self, img, row, col):
        if row == img.shape[0] or col == img.shape[1] or row < 0 or col < 0:
            return
        
        if img[(row, col)] == 0:
            return
        
        
        img[(row, col)] = 0
        
        LPR.floodfill(self, img, row - 1, col)
        LPR.floodfill(self, img, row, col + 1)
        LPR.floodfill(self, img, row + 1, col)
        LPR.floodfill(self, img, row, col - 1)    
        
    def split(self, original_img, filled_img, whitePixNum=3, isWrite=True, writePath='ROI'):
        indices = []

        for i in range(filled_img.shape[1]):
            if filled_img[:, i].sum() >= 0 and filled_img[:, i].sum() < 255 * whitePixNum:
                indices.append(i)
                
        medians = []
        cluster = []

        for i in range(len(indices)):
            if i > 0 and indices[i] - indices[i - 1] <= 1:
                cluster.append(indices[i])
            else:
                if len(cluster) > 0:
                    median = np.median(cluster)
                    medians.append(median)
                cluster = [indices[i]]

        if len(cluster) > 0:
            medians.append(np.median(cluster))

        medians = np.ceil(medians).astype(int)
        medians = np.ceil(medians * (original_img.shape[1] / self.dim2)).astype(int)
        print(f'MEDIANS of each slicing lines: {medians}')
        
        i = 0
        chars = []
        while True:
            if len(medians) == 0:
                break
            if i == len(medians) - 1:
                temp = original_img[:, medians[len(medians) - 1]:]
                chars.append(temp)
                if isWrite:
                    cv.imwrite(f'{writePath}/ROI_{i}.jpg', temp)
                break
            if i == 0 and medians[0] != 0:
                temp = original_img[:, 0:medians[0]]
                chars.append(temp)
                if isWrite:
                    cv.imwrite(f'{writePath}/ROI_{i}.jpg', temp)
            
            temp = original_img[:, medians[i]:medians[i + 1]]
            chars.append(temp)
            if isWrite:
                cv.imwrite(f'{writePath}/ROI_{i}.jpg', temp)
                i += 1
            
        return chars
    
    def display_img(self, img, window_name):
        cv.imshow(window_name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    

########DETECTION ON IMAGE##########

lpr = LPR('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')
input_image = lpr.read_image('home.jpg')

di = lpr.detect_image(input_image, 25)
lpr.display_img(di, 'Matches')
####################################


########DETECTION ON VIDEO##########

lpr = LPR('cascade_1200pos_2600neg_15stage_72_24/cascade.xml')
kernel = cv.getStructuringElement(cv.MORPH_RECT,(1, 1))
lpr.detect_video('driving4.mp4', kernel, maxSize=(24, 72), isWrite=False, printFPS=True, isDisplay=False)

####################################


###########SPLIT FOR OCR############

lpr = LPR(None)

img = lpr.read_image('sample/111.jpg', isGray=True)
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