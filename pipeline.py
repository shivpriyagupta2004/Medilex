import sys
import os
from PIL import Image, ImageOps, ImageFilter
import pytesseract

from ner.ner import extract_entities
from rag.query import run_query

# Default image path
IMG = "sample_prescription.jpg"

def preprocess_image(img_path):
    """Load and preprocess image for OCR."""
    try:
        img = Image.open(img_path)
        
        # Grayscale
        g = ImageOps.grayscale(img)
        # Auto contrast
        g = ImageOps.autocontrast(g, cutoff=2)
        # Resize if small
        w, h = g.size
        if min(w, h) < 900:
            scale = 900 / min(w, h)
            g = g.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        # Threshold
        g = g.point(lambda p: 255 if p > 160 else 0)
        # Sharpen
        g = g.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=5))
        
        return g
    except Exception as e:
        print(f"❌ Image preprocessing error: {e}")
        return None

def run_pipeline(img_path=IMG, translate=True):
    """
    Run complete MediLex pipeline: OCR → NER → RAG → Translation
    
    Args:
        img_path: Path to prescription image
        translate: Whether to translate to Hindi
    """
    print("\n" + "="*70)
    print("🩺 MEDILEX PIPELINE - COMPLETE ANALYSIS")
    print("="*70 + "\n")
    
    # Step 1: Load and preprocess image
    print("📸 STEP 1: Image Preprocessing")
    print("-" * 70)
    
    if not os.path.exists(img_path):
        print(f"❌ Error: Image file '{img_path}' not found.")
        print("Please provide a valid image path.")
        sys.exit(1)
    
    img = preprocess_image(img_path)
    if img is None:
        sys.exit(1)
    
    print(f"✅ Image loaded and preprocessed: {img_path}")
    print()
    
    # Step 2: OCR
    print("🔍 STEP 2: OCR (Optical Character Recognition)")
    print("-" * 70)
    
    try:
        text = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")
        print(f"✅ OCR completed. Extracted {len(text)} characters.")
        print("\n📄 OCR Output (first 400 chars):")
        print("-" * 70)
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print()
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        sys.exit(1)
    
    # Step 3: Entity Extraction (NER)
    print("\n🧠 STEP 3: Named Entity Recognition (NER)")
    print("-" * 70)
    
    ents = extract_entities(text)
    
    # Medicines
    meds = ents.get("medications", [])
    if meds:
        print(f"💊 Found {len(meds)} medicine(s):\n")
        
        # Calculate column widths
        w1 = max(15, max(len(m.get("name", "")) for m in meds))
        w2 = max(10, max(len(m.get("dose", "")) for m in meds))
        w3 = max(15, max(len(m.get("freq_expanded", "")) for m in meds))
        
        # Print table
        header = f'{"Medicine".ljust(w1)}  {"Dose".ljust(w2)}  {"Frequency".ljust(w3)}'
        print(header)
        print("-" * len(header))
        
        for m in meds:
            freq = m.get("freq_expanded") or m.get("freq", "")
            print(
                f'{m.get("name", "").ljust(w1)}  '
                f'{m.get("dose", "").ljust(w2)}  '
                f'{freq.ljust(w3)}'
            )
        print()
    else:
        print("ℹ️  No medicines detected.\n")
    
    # Symptoms
    symptoms = ents.get("symptoms", [])
    if symptoms:
        print(f"🩺 Detected Symptoms: {', '.join(symptoms)}\n")
    
    # Diet & Lifestyle
    diet = ents.get("diet", [])
    if diet:
        print("🥗 DIET & LIFESTYLE RECOMMENDATIONS:")
        print("-" * 70)
        for i, d in enumerate(diet, 1):
            print(f"{i}. {d}")
        print()
    
    # Step 4: RAG Explanation
    print("\n💡 STEP 4: Medical Knowledge Base Query (RAG)")
    print("-" * 70)
    
    # Generate query
    if meds:
        med_names = ", ".join([m["name"] for m in meds])
        query = f"Explain these medicines in simple terms: {med_names}"
    elif symptoms:
        query = f"Explain these symptoms and provide self-care advice: {', '.join(symptoms)}"
    else:
        query = f"Explain the following in simple patient-friendly terms: {text[:200]}"
    
    print(f"Query: {query[:100]}...\n")
    
    try:
        answer = run_query(query, top_k=3)
        print("📚 EXPLANATION (English):")
        print("-" * 70)
        print(answer)
        print()
    except Exception as e:
        print(f"❌ RAG Query Error: {e}\n")
        answer = "Knowledge base query failed."
    
    # Step 5: Translation
    if translate:
        print("\n🌐 STEP 5: Hindi Translation")
        print("-" * 70)
        
        try:
            from googletrans import Translator
            translator = Translator()
            
            # Translate the answer
            translation = translator.translate(answer, src="en", dest="hi")
            
            print("📝 अनुवाद (Hindi):")
            print("-" * 70)
            print(translation.text)
            print()
            
        except ImportError:
            print("⚠️  Translation skipped: googletrans not installed")
            print("   Install with: pip install googletrans==4.0.0rc1\n")
        except Exception as e:
            print(f"⚠️  Translation error: {e}\n")
    
    # Summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"📊 Medicines detected: {len(meds)}")
    print(f"🩺 Symptoms detected: {len(symptoms)}")
    print(f"🥗 Diet recommendations: {len(diet)}")
    print("="*70 + "\n")
    
    # Disclaimer
    print("⚠️  MEDICAL DISCLAIMER:")
    print("This is an AI-powered analysis for informational purposes only.")
    print("Always consult a qualified healthcare professional for medical advice.\n")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MediLex Complete Pipeline")
    parser.add_argument(
        "--img",
        default=IMG,
        help=f"Path to prescription image (default: {IMG})"
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip Hindi translation"
    )
    
    args = parser.parse_args()
    
    run_pipeline(args.img, translate=not args.no_translate)

if __name__ == "__main__":
    main()