# ocr_test.py
import pytesseract
from PIL import Image
import sys
from ner.ner import extract_entities   # import moved to the top

IMG = "sample_prescription.jpg"

def main():
    try:
        img = Image.open(IMG)
    except FileNotFoundError:
        print(f"Image file '{IMG}' not found. Put a file named '{IMG}' in this folder.")
        sys.exit(1)

    # Run OCR
    text = pytesseract.image_to_string(img, lang="eng")
    print("=== OCR OUTPUT ===")
    print(text)

    # Run NER on OCR output
    print("=== NER DETECTION ===")
    entities = extract_entities(text)
    print(entities)

if __name__ == "__main__":
    main()
