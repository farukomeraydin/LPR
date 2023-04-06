import cv2 as cv
import numpy as np
from hsvfilter import HsvFilter


class Vision:
    TRACKBAR_WINDOW = "Trackbars"
    
    img_crop = None
    crop_w = 0
    crop_h = 0
    method = None
    
    def __init__(self, img_crop_path, method=cv.TM_CCOEFF_NORMED):
        self.img_crop = cv.imread(img_crop_path, cv.IMREAD_UNCHANGED)
        self.crop_w = self.img_crop.shape[1]
        self.crop_h = self.img_crop.shape[0]
        self.method = method #'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'

    def find(self, img, threshold=0.5, max_results=10):
        result = cv.matchTemplate(img, self.img_crop, self.method) 
        
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))
        #print(locations)
        if not locations:
            return np.array([], dtype=np.int32).reshape(0, 4)
        
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.crop_w, self.crop_h]
            rectangles.append(rect)
            rectangles.append(rect)
        
        rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
        if len(rectangles) > max_results:
            print('Warning: too many results, raise the threshold.')
            rectangles = rectangles[:max_results]
        #print(rectangles)
        
        return rectangles
        
    
    def get_click_points(self, rectangles):
        points = []          
        for (x, y, w, h) in rectangles:
            center_x = x + int(w / 2)
            center_y = y + int(h / 2)
                
            points.append((center_x, center_y))
        return points
    
    def draw_rectangles(self, img, rectangles):
        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        for (x, y, w, h) in rectangles:
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv.rectangle(img, top_left, bottom_right, line_color, lineType=line_type)
             
        return img
    
    def draw_crosshairs(self, img, points):   
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS
        for (center_x, center_y) in points:
            cv.drawMarker(img, (center_x, center_y), marker_color, marker_type)
        return img
    
    def init_control_gui(self):
        cv.namedWindow(self.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)
        
        def nothing(position):
            pass
        
        cv.createTrackbar('HMin', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        
        
        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, 255)
        
        cv.createTrackbar('SAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        
    def get_hsv_filter_from_controls(self):
        hsv_filter = HsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', self.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACKBAR_WINDOW)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACKBAR_WINDOW)
        return hsv_filter
        
    def apply_hsv_filter(self, original_image, hsv_filter=None):
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
        
        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()
            
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, hsv_filter.vSub)
        hsv = cv.merge([h, s, v])
            
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])
        
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)
        
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)
        
        return img
    
    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
            
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c
        