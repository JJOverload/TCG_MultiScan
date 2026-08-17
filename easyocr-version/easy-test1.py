import cv2 as cv
import easyocr
# For timer
import datetime
# For drawing bbox
import numpy as np
# For splitext
import os

def bbox_draw(image, bbox_points):
    clean_points = np.array(bbox_points, dtype=np.int32)

    cv.polylines(
        image,
        [clean_points],
        isClosed=True,
        color=(0,255,0),
        thickness=2
        )


reader = easyocr.Reader(["en"], gpu=False)
#results = reader.readtext("card.jpg")
scan_names = ["1_python-ocr.jpg", "FyndhornElves1.jpg", "CardPileSample1.jpg", "tegwyll-nonlands-Copy.jpg"]

for name in scan_names:
    print("Starting the timer now for image:", name)
    start_time = datetime.datetime.now()

    image = cv.imread(name)
    results = reader.readtext(
                    image,
                    slope_ths=0.5,
                    ycenter_ths=1.0, #default
                    height_ths=1.0, #not default
                    width_ths=1.0
                    )
    for bbox, text, confidence in results:
        '''
        print("-----TEST START-----")
        print(type(bbox))
        print(np.array(bbox).dtype)
        print(bbox)
        print("-----TEST END-----")
        '''
        print("Text:", text)
        print("Confidence:", confidence)
        print("Bounding Boxes:", bbox)
        bbox_draw(image, bbox)
        print("--------------------------------------------")

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    print(f"Elapsed time for {name} is {elapsed_time.total_seconds():.2f} seconds.")
    
    basename, extension = os.path.splitext(name) #splitext helps separate the extension away from the basename
    cv.imwrite(basename+"-modified"+extension, image)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")