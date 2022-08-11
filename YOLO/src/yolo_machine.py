from ast import Constant, expr_context
from codecs import escape_encode
from concurrent.futures.process import EXTRA_QUEUED_CALLS
from glob import escape
from tkinter.tix import MAIN
from turtle import color, width
import pandas as pd
from pip import main
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from error import ERRORCODE
from src.image_process import imageProc

DEBUG = 1

BLOB_FAST_SIZE = (320, 320)
BLOB_NONAL_SIZE = (416, 416)
BLOB_SLOW_SIZE = (609, 609)
BLOB_MEAN = (0, 0, 0)
BLOB_SCALE_FACTOR = 1 / 256

class machine:
    def __init__(self) -> None:
        self.yolo_image = None
        self.layer_names = None
        self.output_layers = None
        self.indexes = None
        self.class_ids = []
        self.confidences = []
        self.boxes = []

        improc = imageProc()

    def yolo_init(self):
        try:
            self.log("yolo init...")
            # make neural network from model file and config file
            self.net = cv2.dnn.readNet("YOLO/model/yolov3.weights", "YOLO/cfg/yolov3.cfg")
            # get the object name from name file
            with open("YOLO/cfg/coco.names", 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.layer_names = self.net.getLayerNames()
            # self.log(self.layer_names)
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except Exception as e:
            print("YOLO_INIT FUNC ERR {0}".format(e))

    def yolo_run(self, img):
        try:
            self.log("yolo running...")
            self.yolo_image = improc.loadImage(img)
            self.yolo_image = improc.setImageRGB(self.yolo_image)
            height, width, channels = self.yolo_image.shape

            # convert image to 4D matrix object
            blob = cv2.dnn.blobFromImage(img, BLOB_SCALE_FACTOR, BLOB_NONAL_SIZE, BLOB_MEAN, swapRB=True, crop=False)
            # input the object to neural network
            self.net.setInput(blob)
            # run the neural network forward
            outs = self.net.forward(self.output_layers)

            # show the infomation to screen
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # coordinate
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        self.boxes.append([x,y,w,h])
                        self.confidences.append(float(confidence))
                        self.class_ids.append(class_id)
            
            self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)

        except Exception as e:
            print("YOLO_RUN FUNC ERR {0}".format(e))

    def getClassIdx(self):
        try:
            if (len(self.class_ids) > 0):
                return self.class_ids
            else:
                return ERRORCODE._FAIL
        except Exception as e:
            print("GETCLASSIDX FUNC ERR {0}".format(e))

    def getConfid(self):
        try:
            if (len(self.confidences) > 0):
                return self.confidences
            else:
                return ERRORCODE._FAIL
        except Exception as e:
            print("GETCONFIG FUNC ERR {0}".format(e))
    
    def getBoxInfo(self):
        try:
            if (len(self.boxes) > 0):
                return self.boxes
            else:
                return ERRORCODE._FAIL
        except Exception as e:
            print("GETBOXINFO FUNC ERR {0}".format(e))
    
    def getIndexes(self):
        try:
            if self.indexes is not None:
                return self.indexes
            else:
                return ERRORCODE._FAIL
        except Exception as e:
            print("GETINDEXES FUNC ERR {0}".format(e))

    
    def log(self, msg):
        if DEBUG:
            print(msg)

if __name__ == '__main__':
    try:
        pass
        test = machine()
        test.yolo_init()
        test.yolo_run('YOLO/contents/training_images/vid_4_10000.jpg')

    except Exception as e:
        print("MAIN FUNC ERR {0}".format(e))