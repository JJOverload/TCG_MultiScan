import easyocr

reader = easyocr.Reader(["en"], gpu=False)

#results = reader.readtext("card.jpg")
results = reader.readtext("1_python-ocr.jpg")

for bbox, text, confidence in results:
    print(text)
    print(confidence)

print("----------------FyndhornElves1.jpg------------------------")

results = reader.readtext("FyndhornElves1.jpg")

for bbox, text, confidence in results:
    print(text)
    print(confidence)



print("----------------CardPileSample1.jpg------------------------")

results = reader.readtext("CardPileSample1.jpg")

for bbox, text, confidence in results:
    print(bbox)
    print(text)
    print(confidence)


print("----------------tegwyll-nonlands-Copy.jpg------------------------")

results = reader.readtext("tegwyll-nonlands-Copy.jpg")

for bbox, text, confidence in results:
    print(text)
    print(confidence)

