import easyocr

reader = easyocr.Reader(["en"])
results = reader.readtext("vision/ocr_test.jpg")

for result in results:
    print(result)