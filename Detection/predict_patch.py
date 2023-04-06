import cv2 as cv
import matplotlib.pyplot as plt
from vision import Vision
from hsvfilter import HsvFilter
from time import time

def patchify(img, width, height):
    temp = []
    i = 0
    k = 0
    while True:
        if img.shape[0] - k < 10:
            break
        while True:
            if img.shape[1] - i < 10:
                break
            temp.append(img[k:k + height, i:i + width, :])
            i += width
        i = 0
        k += height
        
    return temp


def predict_patch(patch_list, ratio, model):
    license_plates = []
    for i in range(len(patch_list)):
        predict_data = cv.cvtColor(patch_list[i], cv.COLOR_BGR2RGB)
        predict_data = predict_data / (predict_data.max() + 0.001)
        #plt.figure()
        #plt.imshow(predict_data)
        #plt.show()
        predict_result = model.predict(predict_data.reshape(1, 80, 100, 3))
        if predict_result > ratio:
            license_plates.append(i)
    return license_plates


def detect(img, indices):
    start_points = []
    end_points = []
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    for index in indices:
        #plt.figure()
        #plt.imshow(img[(index // 9) * 80:(index // 9) * 80 + 80, ((index % 6) + 1) * 100 - 100:((index % 6) + 1) * 100])
        #plt.show()
        start_points.append((((index % 6) + 1) * 100 - 100, (index // 9) * 80))
        end_points.append((((index % 6) + 1) * 100, (index // 9) * 80 + 80))
    return start_points, end_points
        
from tensorflow.keras.models import load_model

model = load_model('hsv_images/models/binary_classification_hsv.h5')

cap = cv.VideoCapture('driving.mp4')
hsv_filter = HsvFilter(0, 0, 115, 179, 255, 255, 0, 0, 20, 0)
vision_lp = Vision('hsv1.jpg')
loop_time = time()

si = 0

while True:
    ret, screenshot = cap.read()
    processed_image = vision_lp.apply_hsv_filter(screenshot, hsv_filter)
    patches = patchify(processed_image, 100, 80)
    
    patches_resized = []
    for i in patches:
        patches_resized.append(cv.resize(i, (100, 80), interpolation=cv.INTER_AREA))
    output_image = screenshot
    
    indices = predict_patch(patches_resized, 0.5, model)
    """
    if cv.waitKey(1) == ord('p'):
    
        indices = predict_patch(patches_resized, 0.5, model)
        output_image = screenshot
                
        if len(indices) != 0:
            s, e = detect(processed_image, indices)
            for i in range(len(e)):
                output_image = cv.rectangle(screenshot, s[i], e[i], (255, 0, 0), 2)
        
            cv.imwrite(f'hsv_images/results/{si}.jpg', output_image)
            si += 1
    """  
    cv.imshow('Matches', processed_image)
    
    print(f'FPS {1 / (time() - loop_time)}')
    loop_time = time()
    
    if cv.waitKey(1) == ord('q'):
        break
    

cap.release()
cv.destroyAllWindows()

print('Done.')
