import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('dataset2/8.jpg')
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3, 3), 0)
thresh, im_bw = cv2.threshold(blur, np.mean(blur), np.max(blur), cv2.THRESH_BINARY) 
canny = cv2.Canny(im_bw, 150, 255, 1)

plt.figure(figsize=(12, 10))
plt.imshow(blur, cmap='gray')
plt.show()

plt.figure(figsize=(12, 10))
plt.imshow(canny, cmap='gray')
plt.show()

plt.figure(figsize=(12, 10))
plt.imshow(im_bw, cmap='gray')
plt.show()


cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

cnts_index = np.argsort([cv2.boundingRect(i)[0] for i in cnts])

min_area = 100
image_number = 0
for c in cnts_index:
    area = cv2.contourArea(cnts[c])
    
    if area > min_area:
        x,y,w,h = cv2.boundingRect(cnts[c])
        cv2.rectangle(gray, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        
        
        cv2.imwrite(f"ROI_{image_number}.png", ROI)
        image_number += 1

plt.figure(figsize=(12, 10))
plt.imshow(gray, cmap='gray')
plt.show()
