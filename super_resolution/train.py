#!/usr/bin/env python
# coding: utf-8

# In[2]:


from ctypes import resize
import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand
from Subpixel import Subpixel
from DataGenerator import DataGenerator
from PIL import Image
from skimage.transform import pyramid_reduce
import tensorflow as tf



# # In[3]:


base_path = '/Users/jis/car_processed'

x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

print(len(x_train_list))
print(x_train_list[0])

# # # # In[4]:


x1 = np.load(x_train_list[0])
x2 = np.load(x_val_list[0])

print(x1.shape)

# plt.subplot(1, 2, 1)
# plt.imshow(x1)
# plt.subplot(1, 2, 2)
# plt.imshow(x2)
# plt.show()

# # In[5]:


train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=16, dim=(135,240), n_channels=3, n_classes=None, shuffle=True)

val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=16, dim=(135,240), n_channels=3, n_classes=None, shuffle=False)

 # In[6]:


upscale_factor = 8

inputs = Input(shape=(135, 240, 3))

net = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
net = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Conv2D(filters=upscale_factor**2, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(net)
outputs = Activation('relu')(net)

# tf.config.threading.set_intra_op_parallelism_threads(2)
# tf.config.threading.set_inter_op_parallelism_threads(2)
# with tf.device('/CPU:0'):
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse')

# model = load_model('/Users/jis/CPlate/YOLO/lib/super_resolution/models/my_model.h5',custom_objects={'Subpixel':Subpixel})

model.summary()

# # In[7]:

history = model.fit_generator(train_gen, validation_data=val_gen, epochs=1, verbose=1, callbacks=[
    ModelCheckpoint('/Users/jis/CPlate/YOLO/lib/super_resolution/models/eight_scale_model.h5', monitor='val_loss', verbose=1, save_best_only=True)
])

# history  =model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=1, shuffle=False ,callbacks=[
#     ModelCheckpoint('/Users/jis/CPlate/YOLO/lib/super_resolution/models/eight_scale_model.h5', monitor='val_loss', verbose=1, save_best_only=True)
# ])
# # # In[8]:


# x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
# y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))

# print(len(x_test_list), len(y_test_list))
# print(x_test_list[0])

# In[9]:


# test_idx = 24
# downscale = 4

# img = cv2.imread('/Users/jis/Downloads/color_car/test.jpg')
# # img = cv2.imread('/Users/jis/celebA-dataset/img_align_celeba/000001.jpg')

# # # x1_test = np.load(x_test_list[test_idx])
# # # print(x1_test.shape)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # # plt.subplot(1,4,1)
# # # plt.imshow(x1_test)
# # crop = cv2.resize(img, dsize=(384*2, 384*2), interpolation=cv2.INTER_LINEAR)
# crop = cv2.resize(img, dsize=(44, 44), interpolation=cv2.INTER_LINEAR)
# resized = cv2.resize(img, dsize=(174, 174))
# # plt.imshow(crop)
# # plt.show()
# resized = pyramid_reduce(resized, downscale=downscale, multichannel=True)
# # print(resized.shape)
# print(crop.shape)
# norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)
# # # img = cv2.resize(img, dsize=(44,44), interpolation=cv2.INTER_LINEAR)

# # # print(img.shape)
# # cv2.imshow("resized", crop)
# # plt.subplot(1,4,2)
# # plt.imshow(crop)
# # plt.show()
# # x1_test = np.array(resized)
# # # print(x1_test.shape)

# # # x1_test = np.load(x_test_list[test_idx])                             # low quality image
# # # x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True)     # low quality + zoom image
# # # y1_test = np.load(y_test_list[test_idx])                            # answer image
# y_pred = model.predict(norm.reshape((1, 44, 44, 3)))             # model predict image
# # print(y_pred.shape)
# # print(x1_test.shape, y1_test.shape)

# # x1_test = (x1_test * 255).astype(np.uint8)         
# # # x1_test_resized = (x1_test_resized * 255).astype(np.uint8)  
# # # y1_test = (y1_test * 255).astype(np.uint8)                  
# y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)       
# y_pred = cv2.resize(y_pred, dsize=(380, 380))
# # x1_test = cv2.cvtColor(x1_test, cv2.COLOR_BGR2RGB)
# # # x1_test_resized = cv2.cvtColor(x1_test_resized, cv2.COLOR_BGR2RGB)
# # # y1_test = cv2.cvtColor(y1_test, cv2.COLOR_BGR2RGB)
# # y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(15, 10))
# plt.subplot(1, 5, 1)
# plt.title('input')
# plt.imshow(crop)
# # plt.imshow(x1_test)
# plt.subplot(1, 4, 2)
# plt.title('resized')
# plt.imshow(resized)
# plt.subplot(1, 4, 3)
# plt.title('output')
# plt.imshow(y_pred)
# # # plt.subplot(1, 4, 4)
# # # plt.title('groundtruth')
# # # plt.imshow(y1_test)
# plt.show()

# # In[ ]:



