import cv2
import sys
import numpy as np

def pre_process_img(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_image

def get_template(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_image



def match_template():
    nameTemp = get_template("./Templates/name.jpg")
    surnameTemp = get_template("./Templates/surname.jpg")
    dobTemp = get_template("./Templates/dob.jpg")
    img = pre_process_img(sys.argv[1])
    result = cv2.matchTemplate(img, nameTemp, cv2.TM_CCOEFF_NORMED)

    threshold = 0.8
    loc = np.where(result >= threshold)

    for pt in zip(*loc[::-1]):  # Switch columns and rows
        cv2.rectangle(img, pt, (pt[0] + nameTemp.shape[1], pt[1] + nameTemp.shape[0]), (0,0,255), 2)

    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Detected', nameTemp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    match_template()

main()