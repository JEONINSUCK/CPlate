import cv2

from src import img_proc
from src import yolo_machine

COLOR_GREEN = (0,255,0)
COLOR_RED = (255,0,0)
COLOR_BLUE = (0,0,255)
BOX_FONT = cv2.FONT_HERSHEY_PLAIN
BOX_COLOR = COLOR_GREEN
BOX_THICKNESS = 2

# Show to screen from extracted information
def objDetector(img_path):
    try:
        ym = yolo_machine.machine()
        obj_img = img_proc.imageProc()

        ym.yolo_init()
        ym.yolo_run(img_path)
        indexes = ym.getIndexes()
        boxes = ym.getBoxInfo()
        class_ids = ym.getClassIdx()
        confidences = ym.getConfid()
        class_name = ym.getClassName()

        img = obj_img.loadImage(img_path)

        for i in indexes.flatten():
            x, y ,w, h = boxes[i]
            print(x, y, w, h)
            label = str(class_name[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(img, (x,y), ((x+w), (y+h)), BOX_COLOR, BOX_THICKNESS)
            cv2.putText(img, label + " " + confidence, (x, y+70), BOX_FONT, 2, BOX_COLOR, 2)
        
        obj_img.imageShow(img)
        
    except Exception as e:
        print("OBJDETECTOR FUNC ERR {0}".format(e))


path = 'YOLO/contents/training_images/vid_4_11000.jpg'
objDetector(path)
