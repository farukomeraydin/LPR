import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from vision import Vision
from time import time, sleep

from tensorflow.keras.models import load_model

model = load_model('ocr_80.h5')

classes = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
       'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
       'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z']


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
    
    def detect_image(self, input_img, minNeighbors=3, minSize=(dim1, dim2), maxSize=None, isDead=False):
        if isDead:
            interval = input_img.shape[1] // 36
            dead_zone = int(interval * 7.5)
            if len(input_img.shape) > 2:
                input_img = input_img[:, dead_zone:input_img.shape[1] - dead_zone, :]
            else:
                input_img = input_img[:, dead_zone:input_img.shape[1] - dead_zone]
            
        self.rectangles = self.cascade.detectMultiScale(input_img, minNeighbors=minNeighbors, minSize=minSize, maxSize=maxSize)
        detection_image = self.vision.draw_rectangles(input_img.copy(), self.rectangles)
        
        crop = []
        for i in range(len(self.rectangles)):
            x = self.rectangles[i][0]
            y = self.rectangles[i][1]
            w = self.rectangles[i][2]
            h = self.rectangles[i][3]
            crop.append(input_img[y:y + h, x:x + w])
        
        return crop
    
    def get_time_frame(hour=0, min=0, sec=0, frame_rate=None):
        specific_frame = (hour * 3600 + min * 60 + sec) * frame_rate
        return int(specific_frame)
    
    def detect_video(self, video_path, kernel, minNeighbors=3, minSize=(dim1, dim2), maxSize=None, isDisplay=True, printFPS=True, isWrite=True, writePath=None, resize_dim=(1280, 720)):
        cap = cv.VideoCapture(video_path)
        ret, frame = cap.read()
        frame = cv.resize(frame, resize_dim, interpolation = cv.INTER_AREA)
        interval = frame.shape[1] // 36
        dead_zone = int(interval * 7.5)
        
        if isWrite:
            k = 0
            
        if printFPS:
            loop_time = time()
        count = 0
        while True:
            ret, screenshot = cap.read()
            screenshot = cv.resize(screenshot, resize_dim, interpolation = cv.INTER_AREA)
            screenshot = screenshot[:, dead_zone:screenshot.shape[1] - dead_zone, :]
            self.rectangles = self.cascade.detectMultiScale(screenshot, minNeighbors=minNeighbors, minSize=minSize, maxSize=maxSize)
            crops = []
            if len(self.rectangles):
                for i in range(len(self.rectangles)):
                    x = self.rectangles[i][0]
                    y = self.rectangles[i][1]
                    w = self.rectangles[i][2]
                    h = self.rectangles[i][3]
                    if x <= 300 or y <= 300:
                        continue
                    
                    
                    crop = screenshot[y:y + h, x:x + w]
                    crops.append(crop)
                    """
                    #print(f'Crop Shape {i}: {crop.shape}')
                    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                    th1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
                    th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
                    
                    #eroded_image = cv.erode(im_bw, kernel, iterations=1)
                    #dilated_image = cv.dilate(im_bw, kernel, iterations=1)
                    opening = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)
                    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
                    """
                    if isDisplay:
                        plt.figure()
                        plt.imshow(crop, cmap='gray')
                        plt.show()
                    """    
                    if isWrite:
                        cv.imwrite(f'{writePath}/{k}.jpg', closing)
                        cv.imwrite(f'{writePath}/gray_{k}.jpg', gray)
                        k += 1
                    """
            
            if count < 1:
                start = time()
                result = LPR.tryout(self, crops, k=0, maxWhite=1, fillcount=2, upper_crop_size=5, lower_crop_size=5, isDisplay=isDisplay)
                stop = time()
                print(f'TIME: {stop - start}')
                if result is not None:
                    count += 1
            detection_image = self.vision.draw_rectangles(screenshot, self.rectangles)
            
            cv.imshow('Matches', detection_image)
            
            if printFPS:
                print(f'FPS {1 / (time() - loop_time + 0.000001)}')
                loop_time = time()
            
            key = cv.waitKey(1)
            
            if key == ord('q'):
                break
            
        cap.release()
        cv.destroyAllWindows()
    
        
    def display_plt(self, img, title):
        plt.figure(figsize=(12, 10))
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.show()
        
    def inverse_threshold(self, image, resize_dim=(dim2, dim1), th=100, adaptive=True, isDisplay=False):
        inv = cv.bitwise_not(image)
        if isDisplay:
            LPR.display_plt(self, inv, 'INV')

        inv_down = cv.resize(inv, resize_dim, cv.INTER_AREA)
        if isDisplay:
            LPR.display_plt(self, inv_down, 'INV Downscaled')
        
        if adaptive:
            inv_down = cv.adaptiveThreshold(inv_down, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        else:
            inv_down[inv_down >= th] = 255
            inv_down[inv_down < th] = 0
        if isDisplay:
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
        
    def crop(self, img, upper_crop_size=5, lower_crop_size=5):
        return img[upper_crop_size:img.shape[0] - lower_crop_size, :]
        
        
    def split(self, original_img, filled_img, minWhitePix=0, maxWhitePix=3, isWrite=True, writePath='ROI'):
        indices = []

        for i in range(filled_img.shape[1]):
            if filled_img[:, i].sum() >= 255 * minWhitePix and filled_img[:, i].sum() < 255 * maxWhitePix:
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
        print(f'MEDIANS of Downscaled Image: {medians}')
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
                #i += 1
            i += 1
            
        return chars
    
    def display_img(self, img, window_name):
        cv.imshow(window_name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    def foo(self, characters):
        predictions = ''
            
        for i in range(len(characters)):
            characters[i] = cv.resize(characters[i], (80, 200), interpolation = cv.INTER_AREA)
            predict_result = model.predict(characters[i].reshape(1, 200, 80), verbose=0)
            result = np.argmax(predict_result)
            predictions += classes[result]
            
        return predictions
        
    def test(self, di, maxWhite=1, k=0, fillcount=1, upper_crop_size=5, lower_crop_size=5, isDisplay=False):
        if len(di) == 0:
            return None, None
        
        row = [0, 2]
        col = [5, 15]
        if isDisplay:
            LPR.display_plt(self, di[k], 'Matches')
        if len(di[k].shape) > 2:
            di[k] = np.mean(di[k], axis=2).astype(np.uint8)
        if isDisplay:
            LPR.display_plt(self, di[k], 'Matches')
        orig, inv = LPR.inverse_threshold(self, di[k], adaptive=True)
        
        for i in range(fillcount):
            LPR.floodfill(self, inv, row[i], col[i])
        if isDisplay:    
            LPR.display_plt(self, inv, 'filled')
        
        
        inv = LPR.crop(self, inv, upper_crop_size=upper_crop_size, lower_crop_size=lower_crop_size)
        
        if isDisplay:
            LPR.display_plt(self, inv, 'Horizontal Crop')

        character = LPR.split(self, orig, inv, minWhitePix=0, maxWhitePix=maxWhite, isWrite=True)
        
        
        prediction = LPR.foo(self, character)
        
        return character, prediction

    def count(self, prediction):
        c_a = 0
        c_d = 0
        for i in range(len(prediction)):
            if prediction[i].isdigit():
                c_d += 1
            else:
                c_a += 1
                
        return c_a, c_d

    def improve(self, letter_count, digit_count, prediction):
        if len(prediction) >= 4 and  len(prediction) <= 9:
            if prediction[0:3].isdigit():
                prediction = prediction[1:]
                return prediction
            
            if prediction[:2].isdigit() == False and prediction[1].isalpha():
                prediction = '0' + prediction
                if prediction[::-1][0:5].isdigit():
                    prediction = prediction[::-1][1:][::-1]
                return prediction
                
            if prediction[:2].isdigit() == False and prediction[1].isdigit():
                prediction = '0' + prediction[1:]
                if prediction[::-1][0:5].isdigit():
                    prediction = prediction[::-1][1:][::-1]
                return prediction
                
            if prediction[-1].isalpha():
                prediction = prediction[:len(prediction) - 1]
                return prediction
            
            
                   
        else:
            return None

    def tryout(self, di, k, maxWhite, fillcount, upper_crop_size, lower_crop_size, isDisplay=False):
        character, prediction = LPR.test(self, di, k=k, maxWhite=maxWhite, fillcount=fillcount, upper_crop_size=upper_crop_size, lower_crop_size=lower_crop_size)
        if character == None:
            return None
        prediction = prediction.replace(" ", "")
        if isDisplay:
            m = 0
            for i in character:
                LPR.display_plt(self, i, 'Char')
                m += 1
        print(f'PREDICTION:{prediction}')
        
        letter_count, digit_count = LPR.count(self, prediction)
        print(f'Letters: {letter_count}, Digits: {digit_count}')
        
        improved_prediction = LPR.improve(self, letter_count, digit_count, prediction)
        print(f'Improved Prediction:{improved_prediction}')
        
        return 1
