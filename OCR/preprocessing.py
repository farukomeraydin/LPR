import cv2
import matplotlib.pyplot as plt

image_file = "dataset2/4.jpg"
img = cv2.imread(image_file)

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width = im_data.shape[0], im_data.shape[1]
    
    figsize = width / float(dpi), height / float(dpi)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.axis('off')
    
    ax.imshow(im_data, cmap='gray')
    
    plt.show()
    
display(image_file)

inverted_image = cv2.bitwise_not(img)
cv2.imwrite('dataset2/inverted.jpg', inverted_image)

display('dataset2/inverted.jpg')

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(img)
cv2.imwrite('dataset2/gray.jpg', gray_image)

display('dataset2/gray.jpg')

thresh, im_bw = cv2.threshold(gray_image, 100, 150, cv2.THRESH_BINARY)
cv2.imwrite('dataset2/bw_image.jpg', im_bw)

display('dataset2/bw_image.jpg')

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

no_noise = noise_removal(im_bw)
cv2.imwrite('dataset2/no_noise.jpg', no_noise)

display('dataset2/no_noise.jpg')

def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

eroded_image = thin_font(no_noise)
cv2.imwrite('dataset2/eroded_image.jpg', eroded_image)

display('dataset2/eroded_image.jpg')

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

dilated_image = thick_font(no_noise)
cv2.imwrite('dataset2/dilated_image.jpg', dilated_image)

display('dataset2/dilated_image.jpg')

def remove_borders(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntSorted[-1] #Largest
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y + h, x:x + w]
    return crop

no_borders = remove_borders(no_noise)
cv2.imwrite('dataset2/no_borders.jpg', no_borders)
display('dataset2/no_borders.jpg')

color = [255, 255, 255]
top, bottom, left, right = [150] * 4
image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
cv2.imwrite('dataset2/image_with_border.jpg', image_with_border)
display('dataset2/image_with_border.jpg')

