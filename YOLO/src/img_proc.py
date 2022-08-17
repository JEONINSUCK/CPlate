import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

COLOR_GREEN = (0,255,0)
COLOR_RED = (255,0,0)
COLOR_BLUE = (0,0,255)
THICKNESS = 2
IMAGE_QUNTITY = 3

class imageProc:
    def __init__(self) -> None:
        self.csv_image = None
        self.single_image = None
        self.csv_data = None
        self.csv_rows = []
        self.csv_rows_idx = 0
        self.csv_columns = []
        self.csv_columns_idx = 0
        self.foreign_key_idx = 0

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
    
    def imageShow(self, img):
        try:
            plt.imshow(img)
            plt.show()
        except Exception as e:
            print("IMAGESHOW FUNC ERR {0}".format(e))
        

if __name__ == '__main__':
    try:
        improc = imageProc()
        improc.loadCsv('YOLO/contents/train_solution_bounding_boxes.csv')
        improc.csvPaser()
        for i in range(IMAGE_QUNTITY):
            path = 'YOLO/contents/training_images/' + improc.csv_rows[i]
            print(path)
            improc.single_image = improc.loadImage(path)
            improc.single_image = improc.setImageRGB(improc.single_image)
            height, width, channels = improc.single_image.shape
            point = improc.csv_data.iloc[i]
            pt1 = (int(point['xmin']), int(point['ymax']))
            pt2 = (int(point['xmax']), int(point['ymin']))
            cv2.rectangle(improc.single_image, pt1, pt2, color=COLOR_GREEN, thickness=THICKNESS)
            improc.imageShow()
    except Exception as e:
        print("RUN FUNC ERR {0}".format(e))