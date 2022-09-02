import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from skimage.transfrom import pyramid_expand

from Subpixel import Subpixel
from DataGenerator import DataGenerator

# base_path = 'YOLO/contents/sample_submission'
base_path = '/Users/jis/cocodataset'

x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

x1 = np.load(x_train_list[0])
x2 = np.load(x_val_list[0])

print(x1.shape, x2.shape)

plt.subplot(1,2,1)
plt.imshow(x)

def make_low_quality():
    img_sample = cv2.imread(os.path.join(img_base_path, eval_list[0][0]))

    h, w, _ = img_sample.shape

    # make square
    crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2),:]
    # downscale 4 times -> 1/4
    resized_sample = pyramid_reduce(crop_sample, downscale=4)

    pad = int((crop_sample.shape[0] - resized_sample.shape[0]) / 2)
    padded_sample = cv2.copyMakeBorder(resized_sample, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_constant, value=(0,0,0))
    print(crop_sample.shape, padded_sample.shape)
    plt.figure(figsize=(12,5))

def test():
    downscale = 4
    n_train = 162770
    n_val = 19867
    n_test = 19962

    for i, e in enumerate(eval_list):
        # get sample image
        filename, ext = os.path.splitext(e[0])
        img_path = os.path.join(img_base_path, e[0])
        img = cv2.imread(img_path)
        h ,w, _ = img.shape

        # make square
        crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2),:]
        # downscale 4 times -> 1/4
        resized_sample = pyramid_reduce(crop_sample, downscale=downscale)

        # normalize(정규화) image
        norm = cv2.normalize(crop_sample.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

        if int(e[1] == 0):   # train set
            np.save(os.path.join(target_img_path, 'x_train', filename + '.npy'), resized)
            np.save(os.path.join(target_img_path, 'y_train', filename + '.npy'), norm)
        elif int(e[1] == 1):   # validation set
            np.save(os.path.join(target_img_path, 'x_val', filename + '.npy'), resized)
            np.save(os.path.join(target_img_path, 'y_val', filename + '.npy'), norm)
        elif int(e[1] == 2):   # test set
            np.save(os.path.join(target_img_path, 'x_test', filename + '.npy'), resized)
            np.save(os.path.join(target_img_path, 'y_test', filename + '.npy'), norm)

