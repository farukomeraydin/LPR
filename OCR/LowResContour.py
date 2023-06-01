import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.chdir(r'C:\Users\faruk\OneDrive\Masaüstü\haar')


def read_images(path, path_gray):
    image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.imread(path_gray, cv2.COLOR_BGR2GRAY)
    
    return im_gray, image

def get_zeros(img):
    zeros = np.zeros((img.shape[1]), dtype=int)
    nonzeros = np.zeros((img.shape[1]), dtype=int)
    
    for i in range(0, img.shape[1]):
        zeros[i] = (img[:,i] == 0).sum()
        nonzeros[i] = (img[:,i] != 0).sum()
        
    return zeros, nonzeros


def display(img, title):
    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()
    
def get_contours(original, canny, min_area=100):
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts_index = np.argsort([cv2.boundingRect(i)[0] for i in cnts])
    image_number = 0
    for c in cnts_index:
        area = cv2.contourArea(cnts[c])
        
        if area > min_area:
            x,y,w,h = cv2.boundingRect(cnts[c])
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
            ROI = original[y:y+h, x:x+w]
            
            
            cv2.imwrite(f"ROI_{image_number}.jpg", ROI)
            image_number += 1


gray, bw = read_images('sample/109.jpg', 'sample/gray_109.jpg')
display(gray, 'GRAY')
    
inv = cv2.bitwise_not(bw)
display(inv, 'INV')

def morph(kernel, img):
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    eroded_image = cv2.erode(img, kernel, iterations=1)
    dilated_image = cv2.dilate(img, kernel, iterations=1)
    
    return opening, closing, eroded_image, dilated_image

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
op, clo, ero, dil = morph(kernel, inv)

def floodfill(img, row, col):
    if row == img.shape[0] or col == img.shape[1] or row < 0 or col < 0:
        return
    
    if img[(row, col)] == 0:
        return
    
    
    img[(row, col)] = 0
    
    floodfill(img, row - 1, col)
    floodfill(img, row, col + 1)
    floodfill(img, row + 1, col)
    floodfill(img, row, col - 1)
    
inv[inv > 100] = 255
inv[inv < 100] = 0
display(inv, 'Not filled')
floodfill(inv, 2, 10)
display(inv, 'filled')
floodfill(inv, 23, 30)
display(inv, 'filled')

indices = []

for i in range(inv.shape[1]):
    if inv[:, i].sum() >= 0 and inv[:, i].sum() < 255 * 5:
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
print(f'MEDIANS of each slicing lines: {medians}')

i = 0
while True:
    if len(medians) == 0:
        break
    if i == len(medians) - 1:
        temp = gray[:, medians[len(medians) - 1]:]
        cv2.imwrite(f'ROI/ROI_{i}.jpg', temp)
        break
    if i == 0 and medians[0] != 0:
        temp = gray[:, 0:medians[0]]
        cv2.imwrite(f'ROI/ROI_{i}.jpg', temp)
    
    temp = gray[:, medians[i]:medians[i + 1]]
    cv2.imwrite(f'ROI/ROI_{i}.jpg', temp)
    i += 1
