from gettext import find
import matplotlib.pyplot as plt
import pytesseract
import cv2
import imutils
import requests
import numpy as np

# image show function
def plt_imshow(title='image', img=None, figsize=(8,5)):
    # make window
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <=2:
                # convert image gray -> RGB
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                # convert image BGR -> RGB
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
            
            # show many graph to one image
            # subplot(row, colum, index)
            plt.subplot(1, len(img), i+1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else: 
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


def find_edge(img, sigma, minval=75, maxval=200 , width=500):

    if width == None:
        width = 500

    src_img = img.copy()
    src_img = imutils.resize(src_img, width=width)

    # convert the image to grayscale and apply to blur
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5,5), sigma)
    # find the edge
    edged = cv2.Canny(blurred_img, minval, maxval)
    return edged
    # plt_imshow(['gray', 'blurred', 'edged'], [gray_img, blurred_img, edged])

# image load
url = 'https://user-images.githubusercontent.com/69428232/148330274-237d9b23-4a79-4416-8ef1-bb7b2b52edc4.jpg'

# convert the image to array
image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
# convert the binary to image
org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
image = org_image.copy()
image = imutils.resize(image, width=500)
ratio = org_image.shape[1] / float(image.shape[1])
 
# 이미지를 grayscale로 변환하고 blur를 적용
# 모서리를 찾기위한 이미지 연산
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
edged = cv2.Canny(blurred, 75, 200)
# edged_img = find_edge(org_image, sigma=0)

# find contours and sort by size
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

receiptCnt = None

# detect the figure which include 4 edge from sorted contours
for cnt in cnts:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    # judge the first square to receipt because contours sorted by size
    if len(approx) == 4:
        receiptCnt = approx
        break

# if not detect the edge, error process
if receiptCnt is None:
    raise Exception(("Could not find receipt outline."))

output = image.copy()
cv2.drawContours(output, [receiptCnt], -1, (0,255,0), 2)
plt_imshow("Receipt Outline", output)