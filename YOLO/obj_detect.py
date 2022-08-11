import cv2

from src import image_process
from src import yolo_machine


COLOR_GREEN = (0,255,0)
COLOR_RED = (255,0,0)
COLOR_BLUE = (0,0,255)
BOX_FONT = cv2.FONT_HERSHEY_PLAIN
BOX_COLOR = COLOR_GREEN

# Show to screen from extracted information
def objDetector():
    try:
        ym = yolo_machine.machine()
        ym.yolo_init()
        ym.yolo_run('YOLO/contents/training_images/vid_4_10000.jpg')
        indexes = ym.getIndexes()
        boxes = ym.getBoxInfo()
        class_ids = ym.getClassIdx()
        confidence = ym.getConfid()

        for i in indexes.flatten():
            x, y ,w, h = boxes[i]
            print(x, y, w, h)
            label = str(self.classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = BOX_COLOR
            # cv2.rectangle(img, (x,y), ((x+w), (y+h)), color, 2)
            # cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0,255,0), 2)
        
    except Exception as e:
        print("OBJDETECTOR FUNC ERR {0}".format(e))


