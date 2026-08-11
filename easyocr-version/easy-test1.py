import cv2 as cv
import easyocr

reader = easyocr.Reader(["en"], gpu=False)

#results = reader.readtext("card.jpg")

scan_names = ["1_python-ocr.jpg", "FyndhornElves1.jpg", "CardPileSample1.jpg", "tegwyll-nonlands-Copy.jpg"]

for name in scan_names:
    image = cv.imread(name)
    results = reader.readtext(image)
    for bbox, text, confidence in results:
        print("Text:", text)
        print("Confidence:", confidence)
        print("Bounding Boxes:", bbox)
        print("--------------------------------------------")