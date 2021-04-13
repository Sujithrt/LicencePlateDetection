import cv2
import pytesseract
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def extract_num(image_name):
    global read
    yellow_count = 0
    img = cv2.imread(image_name)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 150, 50])
    upper = np.array([35, 255, 255])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in nplate:
        # a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        # plate = img[y+a:y+h-a, x+b:x+w-b]
        plate = img[y:y+h, x:x+w]
        kernels = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernels, iterations=1)
        plate = cv2.erode(plate, kernels, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        plate_HSV = img_HSV[y:y + h, x:x + w]
        mask = cv2.inRange(plate_HSV, lower, upper)
        if cv2.countNonZero(mask) > 0:
            yellow_count += 1
            print('Yellow Number Plate Detected!')
            print('Number of Yellow Number Plates: ', yellow_count)
        else:
            pass
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        print('Number Plate Text: ', read)
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('Plate', plate)
        cv2.waitKey()


    cv2.imshow('Result', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

extract_num('Car Images/Cars60.png')