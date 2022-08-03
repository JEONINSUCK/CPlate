import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os

box = pd.read_csv('YOLO/contents/train_solution_bounding_boxes.csv')

sample = cv2.imread('YOLO/contents/training_images/vid_4_10000.jpg')
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
point = box.iloc[1]
pt1 = (int(point['xmin']), int(point['ymax']))
pt2 = (int(point['xmax']), int(point['ymin']))
cv2.rectangle(sample, pt1, pt2, color=(0,255,0), thickness=2)
plt.imshow(sample)
plt.show()
