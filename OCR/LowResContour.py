import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('dataset2/10.jpg')
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

zeros = np.zeros((im_bw.shape[1]), dtype=int)
nonzeros = np.zeros((im_bw.shape[1]), dtype=int)

for i in range(0, im_bw.shape[1]):
    zeros[i] = (im_bw[:,i] == 0).sum()
    nonzeros[i] = (im_bw[:,i] != 0).sum()

indices = []
for i in range(len(nonzeros)):
    if nonzeros[i] < 30 and nonzeros[i] > 20:
        im_bw[:, i] = 255
        indices.append(i)
        

plt.figure(figsize=(12, 10))
plt.imshow(im_bw, cmap='gray')
plt.show()

nonzeros2 = np.zeros((im_bw.shape[0]), dtype=int)

for i in range(0, im_bw.shape[0]):
    nonzeros2[i] = (im_bw[i,:] != 0).sum()

row1 = 0    
for i in range(len(nonzeros2)):
    if nonzeros2[i] - nonzeros2[0] > 20:
        row1 = i
        break

row2 = len(nonzeros2)
for i in range(len(nonzeros2) - 1, 0, -1):
    if nonzeros2[i] - nonzeros2[len(nonzeros2) - 1] > 20:
        row2 = i
        break        


threshold = 2
medians = []
cluster = []

for i in range(len(indices)):
    if i > 0 and indices[i] - indices[i - 1] <= threshold:
        cluster.append(indices[i])
    else:
        if len(cluster) > 0:
            median = np.median(cluster)
            medians.append(median)
        cluster = [indices[i]]

if len(cluster) > 0:
    median = np.median(cluster)
    medians.append(median)

medians = np.ceil(medians).astype(int)
print(medians)
      

i = 0
while True:
    if i == len(medians) - 1:
        break
    temp = gray[row1:row2, medians[i]:medians[i + 1]]
    cv2.imwrite(f'ROI_{i}.jpg', temp)
    i += 1        

        