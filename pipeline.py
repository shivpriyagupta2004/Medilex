# pipeline.py
import sys
from PIL import Image
import pytesseract

from ner.ner import extract_entities
from rag.query import run_query
from googletrans import Translator

IMG = "sample_prescription.jpg"  # change filename if needed

def run_pipeline(img_path=IMG):
    # --- OCR ---
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"❌ Image file '{img_path}' not found.")
        sys.exit(1)

    text = pytesseract.image_to_string(img, lang="eng")
    print("\n=== OCR OUTPUT ===")
    print(text)

    # --- NER ---
    entities = extract_entities(text)
    print("\n=== NER DETECTION ===")
    print(entities)

    # --- RAG Explanation ---
    query = f"Explain this prescription or symptoms in simple terms:\n{text}"
    answer = run_query(query)
    print("\n=== RAG EXPLANATION (English) ===")
    print(answer)

    # --- Translation (Hindi) ---
    try:
        translator = Translator()
        translation = translator.translate(answer, src="en", dest="hi")
        print("\n=== अनुवाद (Hindi Translation) ===")
        print(translation.text)
    except Exception as e:
        print(f"\n⚠️ Translation skipped: {e}")

if __name__ == "__main__":
    run_pipeline()
