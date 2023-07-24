import cv2
import pytesseract
import sys
from PIL import Image


def main():
    txt = perform_OCR(sys.argv[1])
    print(txt)
    

def preprocess_image(img_path):
    img = cv2.imread(img_path)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(grey_img, (5, 5), 0)

    return blurred_img

def perform_OCR(img_path):
    
    preprocessed_img = preprocess_image(img_path)

    custom_config = r'--oem 3 --psm 6'
    ocr_text = pytesseract.image_to_string(preprocessed_img)

    return ocr_text

def image_to_string(img, lang=None):
    

if __name__ == "__main__":
    main()



