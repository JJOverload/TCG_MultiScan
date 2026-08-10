import easyocr

reader = easyocr.Reader(["en"])

#results = reader.readtext("card.jpg")
results = reader.readtext("1_python-ocr.jpg")

for bbox, text, confidence in results:
    print(text)
    print(confidence)