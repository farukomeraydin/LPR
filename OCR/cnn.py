import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

files = os.listdir('./plaka')

dataset_x = []

for i in files:
    for k in range(len(os.listdir(f'./plaka/{i}'))):
        dataset_x.append(cv2.imread(f'plaka/{i}/{k+1}.png', 0))

plt.figure()
plt.imshow(dataset_x[0], cmap='gray')
plt.show()


for i in range(len(dataset_x)):
    dataset_x[i] = cv2.resize(dataset_x[i], (60, 140), interpolation=cv2.INTER_AREA)
    

dataset_y = np.full((len(dataset_x), 1), ' ')

for i in range(0, 13):
    dataset_y[i] = files[0]
    
for i in range(13, 23):
    dataset_y[i] = files[1]
    
for i in range(23, 34):
    dataset_y[i] = files[2]
    
for i in range(34, 54):
    dataset_y[i] = files[3]
    
for i in range(54, 66):
    dataset_y[i] = files[4]
    
for i in range(66, 81):
    dataset_y[i] = files[5]
    
for i in range(81, 95):
    dataset_y[i] = files[6]
    
for i in range(95, 106):
    dataset_y[i] = files[7]

for i in range(106, 118):
    dataset_y[i] = files[8]
    
for i in range(118, 128):
    dataset_y[i] = files[9]
    
for i in range(128, 142):
    dataset_y[i] = files[10]

for i in range(142, 152):
    dataset_y[i] = files[11]
    
for i in range(152, 162):
    dataset_y[i] = files[12]
    
for i in range(162, 173):
    dataset_y[i] = files[13]
    
for i in range(173, 185):
    dataset_y[i] = files[14]
    
for i in range(185, 197):
    dataset_y[i] = files[15]
    
for i in range(197, 207):
    dataset_y[i] = files[16]
    
for i in range(207, 218):
    dataset_y[i] = files[17]
    
for i in range(218, 228):
    dataset_y[i] = files[18]
    
for i in range(228, 239):
    dataset_y[i] = files[19]
    
for i in range(239, 249):
    dataset_y[i] = files[20]
    
for i in range(249, 259):
    dataset_y[i] = files[21]
    
for i in range(259, 269):
    dataset_y[i] = files[22]
    
for i in range(269, 279):
    dataset_y[i] = files[23]
    
for i in range(279, 289):
    dataset_y[i] = files[24]
    
for i in range(289, 299):
    dataset_y[i] = files[25]
    
for i in range(299, 309):
    dataset_y[i] = files[26]
    
for i in range(309, 319):
    dataset_y[i] = files[27]
    
for i in range(319, 329):
    dataset_y[i] = files[28]
    
for i in range(329, 339):
    dataset_y[i] = files[29]

for i in range(339, 349):
    dataset_y[i] = files[30]
    
for i in range(349, 359):
    dataset_y[i] = files[31]
    
for i in range(359, 369):
    dataset_y[i] = files[32]
    
for i in range(369, 379):
    dataset_y[i] = files[33]
    
for i in range(379, 389):
    dataset_y[i] = files[34]


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dataset_y = le.fit_transform(dataset_y)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from tensorflow.keras.utils import to_categorical

ohe_train_y = to_categorical(train_y)
ohe_test_y = to_categorical(test_y)

train_x = np.array(train_x)
test_x = np.array(test_x)

train_x = train_x / 255
test_x = test_x / 255

train_x = train_x.reshape(-1, 140, 60, 1)
test_x = test_x.reshape(-1, 140, 60, 1)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

model = Sequential(name='OCR')
model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(140, 60, 1), name='Conv2D-1', activation='relu'))
model.add(MaxPooling2D(name='Pooling-1'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), name='Conv2D-2', activation='relu'))
model.add(MaxPooling2D(name='Pooling-2'))
model.add(Flatten(name='Flatten')) 
model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dropout(0.2, name='Dropout-1')) 
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dropout(0.2, name='Dropout-2')) 
model.add(Dense(35, activation='softmax',  name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(train_x, ohe_train_y, epochs=40, batch_size=32, validation_split=0.2)

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Epoch-Categorical Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

eval_result = model.evaluate(test_x, ohe_test_y) 

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

model.save('ocr.h5')

import numpy as np
import glob
for path in glob.glob('test_images/*.jpg'): 
    image_data = plt.imread(path)
    image_data = cv2.resize(image_data, (60, 140), interpolation=cv2.INTER_AREA)
    gray_scaled_image_data = np.average(image_data, axis=2, weights=[0.3, 0.59, 0.11])
    gray_scaled_image_data = gray_scaled_image_data / 255 #Feature scaling
    predict_result = model.predict(gray_scaled_image_data.reshape(1, 140, 60)) 
    result = np.argmax(predict_result)
    print(f'{path}: {classes[result]}')
