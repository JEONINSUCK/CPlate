#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, cv2, glob
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce 
plt.style.use('dark_background')

# In[2]:

# dataset path setting
# base_path = '/Users/jis/celeba-dataset'
# img_base_path = os.path.join(base_path, 'img_align_celeba')
# target_img_path = os.path.join(base_path, 'processed')
base_path = '/Users/jis/'
img_base_path = os.path.join(base_path, 'gray_car')
target_img_path = os.path.join(base_path, 'car_processed')

# print(base_path)
# print(img_base_path)
# print(target_img_path)

# # In[3]:

# call the image name list
# eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=0)
# print(eval_list[0])
# print(eval_list[0][0])
eval_list = []
file_list = os.listdir(os.path.join(base_path, img_base_path))
for file_name in file_list:
    # eval_list.append(os.path.join(base_path, img_base_path, file_name))
    eval_list.append(file_name)
eval_list = np.array(eval_list)
print(eval_list.shape)
print(eval_list[0])



# In[4]:


# img_sample = cv2.imread(os.path.join(base_path, img_base_path, eval_list[0]), cv2.IMREAD_COLOR)
# h, w, _ = img_sample.shape
# img_sample = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)

# # crop a vertical image to a square image
# # crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2), :]


# # zoom a image for down quality
# # resized_sample = pyramid_reduce(crop_sample, downscale=4, channel_axis=True)
# resized_sample = pyramid_reduce(img_sample, downscale=3, multichannel=True)

# pad = int((img_sample.shape[0] - resized_sample.shape[0]) / 2)

# padded_sample = cv2.copyMakeBorder(resized_sample, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

# print(img_sample.shape, padded_sample.shape, resized_sample.shape)

# plt.figure(figsize=(15, 5))
# # plt.subplot(1, 4, 1)
# # plt.imshow(img_sample)
# # plt.subplot(1, 4, 2)
# # plt.imshow(img_sample)
# # plt.subplot(1, 4, 3)
# # plt.imshow(resized_sample)
# # plt.subplot(1, 4, 4)
# # plt.imshow(padded_sample)
# plt.subplot(1,2,1)
# resized_sample = cv2.resize(resized_sample, dsize=(1920, 1080))
# plt.imshow(resized_sample)
# plt.subplot(1,2,2)
# plt.imshow(img_sample)


# plt.show()

# # In[5]:


downscale = 8
n_train = 5370
n_val = 591
n_test = 0

dir_list = [ 'x_train', 'y_train' , 'x_val', 'y_val']
for dirname in dir_list:
    full_dir_name = os.path.join(base_path, target_img_path, dirname)
    if not os.path.exists(full_dir_name):
        os.makedirs(full_dir_name)


for i, e in enumerate(eval_list):
    filename, ext = os.path.splitext(e)
    
    img_path = os.path.join(base_path ,img_base_path, e)
    
    img = cv2.imread(img_path)
    
    h, w, _ = img.shape
    
    # crop = img[int((h-w)/2):int(-(h-w)/2), :]
    # crop = cv2.resize(crop, dsize=(176, 176))
    resized = pyramid_reduce(img, downscale=downscale, multichannel=True)

    norm = cv2.normalize(img.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)
    
    if i<3000:
        np.save(os.path.join(base_path, target_img_path, 'x_train', str(i) + '.npy'), resized)
        np.save(os.path.join(base_path, target_img_path, 'y_train', str(i) + '.npy'), norm)
        print("{0} done.... train img".format(i))
    elif i>3000 and i < 4000:
        np.save(os.path.join(base_path, target_img_path, 'x_val', str(i) + '.npy'), resized)
        np.save(os.path.join(base_path, target_img_path, 'y_val', str(i) + '.npy'), norm)
        print("{0} done.... val img".format(i))
    
    # if int(e[1]) == 0:
    #     np.save(os.path.join(target_img_path, 'x_train', filename + '.npy'), resized)
    #     np.save(os.path.join(target_img_path, 'y_train', filename + '.npy'), norm)
    # elif int(e[1]) == 1:
    #     np.save(os.path.join(target_img_path, 'x_val', filename + '.npy'), resized)
    #     np.save(os.path.join(target_img_path, 'y_val', filename + '.npy'), norm)
    # elif int(e[1]) == 2:
    #     np.save(os.path.join(target_img_path, 'x_test', filename + '.npy'), resized)
    #     np.save(os.path.join(target_img_path, 'y_test', filename + '.npy'), norm)


# In[ ]:



