from itertools import tee
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import cv2
import matplotlib.pyplot as plt




class Denoise:
    def __init__(self) -> None:
        self.x_train = None
        self.x_test = None
    
    def dataLoad(self):
        (self.x_train, _), (self.x_test, _) = fashion_mnist.load_data()

    def dataProcess(self):
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255
        self.x_train = self.x_train[..., tf.newaxis]
        self.x_test = self.x_test[..., tf.newaxis]

        print(self.x_train.shape)
        print(self.x_test.shape)

    def run(self):
        try:
            img = cv2.imread('Image_process/contents/salt&pepper_test.png')
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            plt.subplot(121),plt.imshow(img)
            plt.subplot(122),plt.imshow(dst)
            plt.show()
        except Exception as e:
            print("RUN FUNC ERR {0}".format(e))
if __name__ == '__main__':
    dn = Denoise()
    dn.dataLoad()
    # dn.dataProcess()