import matplotlib.pyplot as plt
import cv2 as cv
"""
count = 0

def patchify(img, width, height):
    i = 0
    k = 0
    global count
    while True:
        if img.shape[0] - k == 0:
            break
        while True:
            if img.shape[1] - i == 0:
                break
            cv.imwrite(f'hsv_images/patches/{count}.jpg', img[k:k + height, i:i + width, :])
            count += 1
            i += width
        i = 0
        k += height

for i in range(1, 368):
    image = cv.imread(f'hsv_images/{i}.jpg')
    full = image[:, :800, :]
    
    full = cv.cvtColor(full, cv.COLOR_BGR2GRAY)
    
    plt.figure()
    plt.imshow(full)
    plt.show()
    
    patches = patchify(full, 100, 80)
"""
import numpy as np

dataset_x = []

for i in range(1, 149):
    image_data = cv.imread(f'hsv_images/{i}.jpg')
    image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)
    image_data = image_data / image_data.max()
    dataset_x.append(image_data)

plt.figure()
plt.imshow(dataset_x[36], cmap='gray')
plt.show()


dataset_y = np.zeros((148, 1), dtype='int8')
for i in range(0, 90):
    dataset_y[i] = 1
    
#np.put(dataset_y, [36, 84, 132, 143, 180, 191, 228, 239, 276, 286, 287, 324, 334, 335, 372, 382, 420, 430, 468, 478, 516, 526, 712, 761, 801, 850, 898, 998, 1000, 1046, 1095, 1138, 1193, ], [1])

size = (100, 80)

resized_dataset_x = []

for i in dataset_x:
    resized_dataset_x.append(cv.resize(i, size, interpolation=cv.INTER_AREA))
    
    
plt.figure()
plt.imshow(resized_dataset_x[36], cmap='gray')
plt.show()


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(resized_dataset_x, dataset_y, test_size=0.2)

train_x = np.array(train_x)
test_x = np.array(test_x)

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

dn121 = DenseNet121(include_top=False, input_shape=(80, 100, 3), weights='imagenet')

result = Flatten(name='Flatten')(dn121.output)
out = Dense(1, activation='sigmoid', name='Output')(result)

model = Model(inputs=dn121.input, outputs=out)

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping('val_loss', patience=5, verbose=1, restore_best_weights=True)

hist = model.fit(train_x, train_y, batch_size=32, epochs=10, validation_split=0.2)

model.save('hsv_images/models/binary_classification_hsv.h5')

import matplotlib.pyplot as plt

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
plt.title('Epoch-Binary Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Binary Accuracy', 'Validation Binary Accuracy'])
plt.show()

eval_result = model.evaluate(test_x, test_y) 

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
    
predict_data = cv.imread('hsv_images/36.jpg')
predict_data = cv.resize(predict_data, size, interpolation=cv.INTER_AREA)
predict_data = cv.cvtColor(predict_data, cv.COLOR_BGR2RGB)
predict_data = predict_data / predict_data.max()

predict_result = model.predict(predict_data.reshape(1, 80, 100, 3))

if predict_result > 0.5:
    print('Positive')
else:
    print('Negative')
    
