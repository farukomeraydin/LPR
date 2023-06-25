import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


files = os.listdir('./mydataset')

dataset_y = []

for i in range(len(files)):
    if files[i].split('_')[0] == 'empty':
        dataset_y.append(' ')
    else:
        dataset_y.append(files[i].split('_')[0])
    
dataset_x = []
for i in files:
    dataset_x.append(cv.imread(f'mydataset/{i}', 0))
    
def display(img, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()
    
for i in range(50, 53):
    display(dataset_x[i], dataset_y[i])
    
print(f'TOTAL DATASET:{len(dataset_y)}')

fig, ax = plt.subplots(figsize =(20, 7))
ax.hist(dataset_y, bins=np.arange(0, 35), width=0.5)
plt.show()

for i in range(len(dataset_x)):
    dataset_x[i] = cv.resize(dataset_x[i], (80, 200), interpolation = cv.INTER_AREA)
    
for i in range(20, 25):
    display(dataset_x[i], dataset_y[i])
    
dataset_x = np.array(dataset_x)

dataset_x = dataset_x / 255

import pandas as pd

ohe_dataset_y = pd.get_dummies(dataset_y)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(dataset_x, ohe_dataset_y, test_size=0.2)

display(train_x[12], train_y.iloc[12][train_y.iloc[12] == 1].index[0])

train_x = train_x.reshape(-1, 200, 80, 1)
test_x = test_x.reshape(-1, 200, 80, 1)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

model = Sequential(name='OCR')
model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(200, 80, 1), name='Conv2D-1', activation='relu'))
model.add(MaxPooling2D(name='Pooling-1'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), name='Conv2D-2', activation='relu'))
model.add(MaxPooling2D(name='Pooling-2'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), name='Conv2D-3', activation='relu'))
model.add(MaxPooling2D(name='Pooling-3'))
model.add(Conv2D(32, (3, 3), strides=(1, 1), name='Conv2D-4', activation='relu'))
model.add(MaxPooling2D(name='Pooling-4'))

model.add(Flatten(name='Flatten')) 

model.add(Dense(256, activation='relu', name='Hidden-1'))
model.add(Dropout(0.2, name='Dropout-1')) 
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dropout(0.2, name='Dropout-2'))
model.add(Dense(64, activation='relu', name='Hidden-3'))
model.add(Dropout(0.2, name='Dropout-3'))
model.add(Dense(64, activation='relu', name='Hidden-4'))
model.add(Dropout(0.2, name='Dropout-4')) 
model.add(Dense(128, activation='relu', name='Hidden-5'))
model.add(Dropout(0.2, name='Dropout-5'))
model.add(Dense(256, activation='relu', name='Hidden-6'))
model.add(Dropout(0.2, name='Dropout-6'))

model.add(Dense(len(np.unique(dataset_y)), activation='softmax',  name='Output'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping('val_loss', patience=15, verbose=1, restore_best_weights=True)

hist = model.fit(train_x, train_y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[esc])

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

eval_result = model.evaluate(test_x, test_y) 

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
    
classes = np.unique(dataset_y)

model.save('ocr_80.h5', save_format='h5')

from tensorflow.keras.models import load_model

model = load_model('ocr_80.h5')

import glob
for path in glob.glob('test_images/*.jpg'): 
    image_data = plt.imread(path)
    image_data = cv.resize(image_data, (80, 200), interpolation=cv.INTER_AREA)
    #gray_scaled_image_data = np.mean(image_data, axis=2)
    gray_scaled_image_data = image_data / 255 #Feature scaling
    predict_result = model.predict(gray_scaled_image_data.reshape(1, 200, 80))
    result = np.argmax(predict_result)
    print(f'{path}: {classes[result]}')
    

predictions = []
y_test = []

for i in range(len(test_x)):
    predict_result = model.predict(test_x[i].reshape(1, 200, 80))
    result = np.argmax(predict_result)
    predictions.append(classes[result])
    y_test.append(test_y.iloc[i][test_y.iloc[i] == 1].index[0])
    print(f'Real:{test_y.iloc[i][test_y.iloc[i] == 1].index[0]} Predicted:{classes[result]}')

    
from sklearn.metrics import confusion_matrix

result = confusion_matrix(y_test, predictions , normalize='pred', labels=classes)

import seaborn as sn
plt.figure(figsize=(24, 24))
plt.title('X:Predicted Values Y:True Values')
sn.set(font_scale=1.4) 
sn.heatmap(result, annot=True, annot_kws={"size": 16}, xticklabels=np.unique(dataset_y), yticklabels=np.unique(dataset_y)) 
plt.show()