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

COLOR_GREEN = (0,255,0)
COLOR_RED = (255,0,0)
COLOR_BLUE = (0,0,255)
BLOB_FAST_SIZE = (320, 320)
BLOB_NONAL_SIZE = (416, 416)
BLOB_SLOW_SIZE = (609, 609)
THICKNESS = 2
IMAGE_QUNTITY = 3


class objdetector:
    def __init__(self) -> None:
        self.csv_image = None
        self.single_image = None
        self.csv_data = None
        self.csv_rows = []
        self.csv_rows_idx = 0
        self.csv_columns = []
        self.csv_columns_idx = 0
        self.foreign_key_idx = 0
        self.layer_names = None
        self.output_layers = None

        self.yolo_init()
    
    def run(self):
        try:
            od.loadCsv('YOLO/contents/train_solution_bounding_boxes.csv')
            od.csvPaser()
            for i in range(IMAGE_QUNTITY):
                path = 'YOLO/contents/training_images/' + self.csv_rows[i]
                print(path)
                self.single_image = self.loadImage(path)
                self.single_image = self.setImageRGB(self.single_image)
                height, width, channels = self.single_image.shape
                point = self.csv_data.iloc[i]
                pt1 = (int(point['xmin']), int(point['ymax']))
                pt2 = (int(point['xmax']), int(point['ymin']))
                cv2.rectangle(self.single_image, pt1, pt2, color=COLOR_GREEN, thickness=THICKNESS)
                self.imageShow()
        except Exception as e:
            print("RUN FUNC ERR {0}".format(e))

    def yolo_init(self):
        try:
            # make neural network from model file and config file
            print("yolo init start...")
            net = cv2.dnn.readNet("YOLO/model/yolov3.weights", "YOLO/cfg/yolov3.cfg")
            classes = []
            with open("YOLO/cfg/coco.names", 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            self.layer_names = net.getLayerNames()
            print(net.getUnconnectedOutLayers())
            self.output_layers = [self.layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            
            img = self.loadImage('YOLO/contents/training_images/vid_4_10000.jpg')
            img = self.setImageRGB(img)
            height, width, channels = img.shape

            # convert image to 4D matrix object
            blob = cv2.dnn.blobFromImage(img, 1/256, BLOB_NONAL_SIZE, (0,0,0), swapRB=True, crop=False)
            # input the object to neural network
            net.setInput(blob)
            # run the neural network forward
            outs = net.forward(self.output_layers)

            # show the infomation to screen
            class_ids = []
            confidences = []
            boxes = []

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
                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Show to screen from extracted information
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0,255, size=(len(boxes), 3))

            for i in indexes.flatten():
                x, y ,w, h = boxes[i]
                print(x, y, w, h)
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x,y), ((x+w), (y+h)), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0,255,0), 2)
            
            plt.imshow(img)
            plt.show()
        except Exception as e:
            print("YOLO_INIT FUNC ERR {0}".format(e))

    def loadCsv(self, csv_file):
        try:
            self.csv_data = pd.read_csv(csv_file)
        except Exception as e:
            print("LOADCSV FUNC ERR {0}".format(e))

    def csvPaser(self):
        try:
            if self.csv_data is not None :
                for col in self.csv_data.columns:
                    self.csv_columns.append(col)
                    self.csv_columns_idx += 1
                self.setForeignKey()

                for rows in self.csv_data.values:
                    self.csv_rows.append(rows[self.foreign_key_idx])
                    self.csv_rows_idx += 1
            else:
                print("Please call the loadCsv() func")
        except Exception as e:
            print("CSVPARSER FUNC ERR {0}".format(e))
    
    def setForeignKey(self, key_name="image"):
        try:
            key = key_name.lower()
            self.foreign_key_idx = self.csv_columns.index(key)
        except Exception as e:
            print("SETFOREIGNKEY FUNC ERR {0}".format(e))
    
    def printCol(self):
        print("columns: {0}".format(self.csv_columns))
        print("index: {0}".format(self.csv_columns_idx))

    def printRow(self):
        print("rows: {0}".format(self.csv_rows))
        print("index: {0}".format(self.csv_rows_idx))

    def loadImageCsv(self, img_path):
        try:
            self.csv_image = pd.read_csv(img_path)
        except Exception as e:
            print("LOADIMAGECSV FUNC ERR {0}".format(e))

    def loadImage(self, img_path):
        try:
            return cv2.imread(img_path)
        except Exception as e:
            print("LOADIMAGE FUNC ERR {0}".format(e))

    def setImageRGB(self, img):
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("SETIMAGERGB FUNC ERR {0}".format(e))
    
    def setImageGray(self, img):
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print("SETIMAGEGRAY FUNC ERR {0}".format(e))
    
    def setImageHSV(self, img):
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        except Exception as e:
            print("SETIMAGEHSV FUNC ERR {0}".format(e))
    
    def setImageLab(self, img):
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        except Exception as e:
            print("SETIMAGELAB FUNC ERR {0}".format(e))
    
    def imageShow(self):
        try:
            plt.imshow(self.single_image)
            plt.show()
        except Exception as e:
            print("IMAGESHOW FUNC ERR {0}".format(e))


if __name__ == '__main__':
    try:
        od = objdetector()
        od.yolo_init()

    except Exception as e:
        print("MAIN FUNC ERR {0}".format(e))