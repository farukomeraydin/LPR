import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

"""
cap = cv.VideoCapture('driving.mp4')

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

def display(disp_type, img, window_size_scale):
    image = None
    if not isinstance(disp_type, str):
        print('First argument must be a string')
    
    cv.namedWindow(disp_type, cv.WINDOW_NORMAL)
    cv.resizeWindow(disp_type, int(width / window_size_scale), int(height / window_size_scale))
    if disp_type == 'gray':
        image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif disp_type == 'rgb':
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif disp_type == 'hsv':
        image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    elif disp_type == 'bgr':
        image = img
    
    if image is not None:
        cv.imshow(disp_type, image)
    else:
        cap.release()
        cv.destroyAllWindows()
        raise ValueError('Options: rgb, bgr, hsv, gray')
"""

df_train = np.loadtxt('security/LPR.v6i.tensorflow/train/_annotations.csv', delimiter=',', skiprows=1, dtype=str, usecols=[0])
df_test = np.loadtxt('security/LPR.v6i.tensorflow/test/_annotations.csv', delimiter=',', skiprows=1, dtype=str, usecols=[0])

fs_train = []
fs_test = []

for i in range(len(df_train)):
    fs_train.append(df_train[:][i].split('_')[0])

train_x = []

for i in fs_train:
    n = cv.imread(f'security/LPR.v6i.tensorflow/train/{i}.jpg')
    n = cv.cvtColor(n, cv.COLOR_BGR2RGB)
    train_x.append(n)

test_x = []

for i in range(len(df_test)):
    fs_test.append(df_test[:][i].split('_')[0])

for i in fs_test:
    n = cv.imread(f'security/LPR.v6i.tensorflow/test/{i}.jpg')
    n = cv.cvtColor(n, cv.COLOR_BGR2RGB)
    test_x.append(n)

train_x = np.array(train_x, dtype='float32')
test_x = np.array(test_x, dtype='float32')    
    
df_train = np.loadtxt('security/LPR.v6i.tensorflow/train/_annotations.csv', delimiter=',', skiprows=1, dtype=np.float16, usecols=[4, 5, 6, 7])
df_test = np.loadtxt('security/LPR.v6i.tensorflow/test/_annotations.csv', delimiter=',', skiprows=1, dtype=np.float16, usecols=[4, 5, 6, 7])

train_y = df_train
test_y = df_test

width = train_x[0].shape[1] #1920
height = train_x[0].shape[0] #1080

for i in range(len(train_x)):
    train_x[i] = train_x[i] / train_x[i].max()
    
for i in range(len(test_x)):
    test_x[i] = test_x[i] / test_x[i].max()

for i in range(len(train_y)):
    train_y[i][::2] = train_y[i][::2] / width
    train_y[i][1::2] = train_y[i][1::2] / height
    
for i in range(len(test_y)):
    test_y[i][::2] = test_y[i][::2] / width
    test_y[i][1::2] = test_y[i][1::2] / height

#from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
"""
#rn101 = ResNet101(include_top=False, input_shape=(640, 640, 3), weights='imagenet')
from tensorflow.keras.applications import MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(1080, 1920, 3))

result = Flatten(name='Flatten')(baseModel.output)
out1 = Dense(1, activation='linear', name='Output-1')(result)
out2 = Dense(1, activation='linear', name='Output-2')(result)
out3 = Dense(1, activation='linear', name='Output-3')(result)
out4 = Dense(1, activation='linear', name='Output-4')(result)
"""

from tensorflow.keras.applications import VGG16

vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

flatten = vgg.output
flatten = Flatten()(flatten)
bboxHead = Dense(64, activation="relu")(flatten)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(16, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="linear")(bboxHead)

model = Model(inputs=vgg.input, outputs=bboxHead)

#model = Model(inputs=baseModel.input, outputs=[out1, out2, out3, out4])
model.compile(optimizer='adam', loss='mse')
model.summary()
#model.compile(optimizer='rmsprop', loss={'Output-1': 'mae', 'Output-2': 'mae', 'Output-3': 'mae', 'Output-4': 'mae'}, loss_weights={'Output-1': 1, 'Output-2': 1, 'Output-3': 1, 'Output-4': 1}, metrics={'Output-1': ['mae'], 'Output-2': ['mae'], 'Output-3': ['mae'], 'Output-4': ['mae']})


from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping('val_loss', patience=10, verbose=1, restore_best_weights=True)

#hist = model.fit(train_x, [train_y[:, 0], train_y[:, 1], train_y[:, 2], train_y[:, 3]], batch_size=2, epochs=100,  validation_split=0.2)
hist = model.fit(train_x, train_y, batch_size=32, epochs=40, callbacks=[esc], validation_split=0.2)
model.save('vgg16.h5')

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
plt.title('Epoch-Mean Absolute Error Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['mse'])
plt.plot(hist.epoch, hist.history['val_mse'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()


from tensorflow.keras.models import load_model

model = load_model('vgg16.h5')

eval_result = model.evaluate(test_x, test_y) 

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

import numpy as np
import glob

for path in glob.glob('security/LPR.v6i.tensorflow/test/*.jpg'): 
    image_data = cv.imread(path)
    image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)
    image_data = image_data / image_data.max()
    predict_result = model.predict(image_data.reshape(1, 224, 224, 3)) 
    print(f'{path}: {predict_result}')
    print(predict_result[0][0] * width, predict_result[0][2] * width, predict_result[0][1] * height, predict_result[0][3] * height)

"""
while True:
    ret, frame = cap.read()
    
    display('bgr', frame, 0.8)

    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()
"""
