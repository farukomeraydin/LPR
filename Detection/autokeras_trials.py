import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

df_train = np.loadtxt('security/LPR.v4i.tensorflow/train/_annotations.csv', delimiter=',', skiprows=1, dtype=np.int16, usecols=[4, 5, 6, 7])
df_test = np.loadtxt('security/LPR.v4i.tensorflow/test/_annotations.csv', delimiter=',', skiprows=1, dtype=np.int16, usecols=[4, 5, 6, 7])

train_x = []
fs = ['f8', 'f13', 'f11', 'f3', 'f40', 'f37', 'f39', 'f22', 'f17', 'f5', 'f19', 'f6', 'f25', 'f38', 'f28', 'f32', 'f7', 'f30', 'f10', 'f14', 'f31', 'f35', 'f15', 'f2', 'f26', 'f41']

for i in fs:
    n = cv.imread(f'security/LPR.v4i.tensorflow/train/{i}.jpg')
    train_x.append(n)

train_y = df_train

test_x = []

fs = ['f23', 'f16', 'f24', 'f4', 'f1', 'f20', 'f21', 'f42', 'f34', 'f12', 'f33', 'f29']

for i in fs:
    n = cv.imread(f'security/LPR.v4i.tensorflow/test/{i}.jpg')
    test_x.append(n)
    
test_y = df_test

train_x = np.array(train_x)
test_x = np.array(test_x)


import autokeras as ak

ir = ak.ImageRegressor(output_dim=4, max_trials=20)

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping('val_loss', patience=10, verbose=1, restore_best_weights=True)

hist = ir.fit(train_x, train_y, batch_size=2, epochs=20, callbacks=[esc], validation_split=0.2)

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, len(hist.epoch), 20))

plt.plot(hist.epoch, hist.history['loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Epoch-Mean Absolute Error Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, len(hist.epoch), 20))

plt.plot(hist.epoch, hist.history['mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()

model = ir.export_model()

eval_result = model.evaluate(test_x, test_y)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
    
model.summary()
model.save('image-regressor-best-model.h5')
