import sys
import os
import io
import argparse
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from ner.ner import extract_entities

DEFAULT_IMG = "sample_prescription.jpg"

def preprocess(img, handwritten=False, verbose=False):
    """
    Preprocess image for better OCR results.
    
    Args:
        img: PIL Image object
        handwritten: Whether the prescription is handwritten
        verbose: Print preprocessing steps
        
    Returns:
        Preprocessed PIL Image
    """
    if verbose:
        print("🔄 Preprocessing image...")
    
    # Convert to grayscale
    g = ImageOps.grayscale(img)
    if verbose:
        print("  ✓ Converted to grayscale")
    
    # Auto contrast
    g = ImageOps.autocontrast(g, cutoff=2)
    if verbose:
        print("  ✓ Applied auto contrast")
    
    # Resize if too small
    w, h = g.size
    if min(w, h) < 900:
        scale = max(1.5, 900 / min(w, h))
        new_size = (int(w * scale), int(h * scale))
        g = g.resize(new_size, Image.Resampling.LANCZOS)
        if verbose:
            print(f"  ✓ Resized to {new_size}")
    
    # Different processing for handwritten vs printed
    if handwritten:
        # Median filter to reduce noise
        g = g.filter(ImageFilter.MedianFilter(size=3))
        # More aggressive threshold for handwritten
        g = g.point(lambda p: 255 if p > 180 else 0)
        if verbose:
            print("  ✓ Applied handwritten-specific processing")
    else:
        # Standard threshold for printed text
        g = g.point(lambda p: 255 if p > 160 else 0)
        if verbose:
            print("  ✓ Applied binary threshold")
    
    # Sharpen the image
    g = g.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=5))
    if verbose:
        print("  ✓ Sharpened image")
    
    return g

def tesseract_ocr(img, handwritten=False, verbose=False):
    """
    Perform OCR on preprocessed image.
    
    Args:
        img: PIL Image object
        handwritten: Whether the prescription is handwritten
        verbose: Print OCR details
        
    Returns:
        Extracted text as string
    """
    # Configure Tesseract
    base_psm = 4 if handwritten else 6
    config = f"--oem 3 --psm {base_psm}"
    
    if verbose:
        try:
            ver = pytesseract.get_tesseract_version()
            print(f"[INFO] Tesseract version: {ver}")
        except Exception:
            print("[WARN] Could not determine Tesseract version")
    
    try:
        if verbose:
            print(f"🔍 Running OCR (PSM mode: {base_psm})...")
        
        text = pytesseract.image_to_string(img, lang="eng", config=config)
        
        if verbose:
            print(f"✅ OCR complete. Extracted {len(text)} characters.")
        
        return text
        
    except Exception as e:
        print(f"❌ ERROR: OCR failed: {e}")
        return ""

def print_table(rows, title="Medicines"):
    """
    Print a formatted table of medicines.
    
    Args:
        rows: List of medicine dictionaries
        title: Table title
    """
    if not rows:
        print(f"\n{'='*60}")
        print(f"No {title.lower()} detected.")
        print(f"{'='*60}\n")
        return
    
    # Calculate column widths
    w1 = max(12, max(len(r.get("name", "")) for r in rows))
    w2 = max(8, max(len(r.get("dose", "")) for r in rows))
    w3 = max(10, max(len(r.get("freq_expanded", r.get("freq", ""))) for r in rows))
    w4 = max(10, max(len(r.get("route", "")) for r in rows))
    
    # Print table
    print(f"\n{'='*60}")
    print(f"{title.upper()}")
    print(f"{'='*60}")
    
    header = (
        f'{"Medicine".ljust(w1)}  '
        f'{"Dose".ljust(w2)}  '
        f'{"Frequency".ljust(w3)}  '
        f'{"Route".ljust(w4)}'
    )
    print(header)
    print("-" * len(header))
    
    for r in rows:
        freq = r.get("freq_expanded") or r.get("freq", "")
        print(
            f'{r.get("name", "").ljust(w1)}  '
            f'{r.get("dose", "").ljust(w2)}  '
            f'{freq.ljust(w3)}  '
            f'{r.get("route", "").ljust(w4)}'
        )
    
    print(f"{'='*60}\n")

def save_output(text, entities, output_dir="output"):
    """
    Save OCR and NER results to files.
    
    Args:
        text: OCR extracted text
        entities: Extracted entities dictionary
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save OCR text
    ocr_file = os.path.join(output_dir, "ocr_output.txt")
    with io.open(ocr_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Saved OCR text to: {ocr_file}")
    
    # Save structured data
    import json
    ner_file = os.path.join(output_dir, "entities.json")
    with open(ner_file, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved entities to: {ner_file}")
    
    # Save medicines as CSV
    meds = entities.get("medications", [])
    if meds:
        csv_file = os.path.join(output_dir, "medicines.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("Medicine,Dose,Frequency,Route\n")
            for m in meds:
                freq = m.get("freq_expanded") or m.get("freq", "")
                f.write(f'"{m.get("name", "")}","{m.get("dose", "")}","{freq}","{m.get("route", "")}"\n')
        print(f"[INFO] Saved medicines CSV to: {csv_file}")

def main():
    """Main function for OCR testing."""
    ap = argparse.ArgumentParser(
        description="OCR and NER extraction from prescription images"
    )
    ap.add_argument(
        "--img",
        default=DEFAULT_IMG,
        help=f"Path to prescription image (default: {DEFAULT_IMG})"
    )
    ap.add_argument(
        "--handwritten",
        action="store_true",
        help="Use handwritten mode for preprocessing"
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed processing information"
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="Save outputs to files"
    )
    ap.add_argument(
        "--preview",
        type=int,
        default=600,
        help="Number of characters to preview from OCR (default: 600)"
    )
    
    args = ap.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.img):
        print(f"❌ ERROR: Image file '{args.img}' not found.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("MEDILEX OCR & NER EXTRACTION")
    print("="*60)
    print(f"Input file: {args.img}")
    print(f"Mode: {'Handwritten' if args.handwritten else 'Printed'}")
    print("="*60 + "\n")
    
    # Load and preprocess image
    try:
        img = Image.open(args.img)
        print(f"✅ Loaded image: {img.size[0]}x{img.size[1]} pixels")
    except Exception as e:
        print(f"❌ ERROR: Could not load image: {e}")
        sys.exit(1)
    
    proc = preprocess(img, args.handwritten, args.verbose)
    
    # Perform OCR
    text = tesseract_ocr(proc, args.handwritten, args.verbose)
    
    if not text.strip():
        print("❌ ERROR: No text extracted from image.")
        sys.exit(1)
    
    # Display OCR output
    print("\n" + "="*60)
    print("OCR OUTPUT (Preview)")
    print("="*60)
    preview_text = text[:args.preview]
    if len(text) > args.preview:
        preview_text += "\n... (truncated)"
    print(preview_text)
    print("="*60)
    
    # Extract entities
    print("\n🧠 Extracting entities...")
    ents = extract_entities(text) or {}
    
    # Display structured data
    print("\n" + "="*60)
    print("NER EXTRACTION RESULTS")
    print("="*60)
    
    # Medicines
    meds = ents.get("medications", [])
    print_table(meds, "Medicines")
    
    # Symptoms
    symptoms = ents.get("symptoms", [])
    if symptoms:
        print("🩺 SYMPTOMS DETECTED:")
        for s in symptoms:
            print(f"  • {s}")
        print()
    
    # Diet advice
    diet = ents.get("diet", [])
    if diet:
        print("🥗 DIET & LIFESTYLE ADVICE:")
        for i, d in enumerate(diet, 1):
            print(f"  {i}. {d}")
        print()
    
    # Save outputs if requested
    if args.save:
        save_output(text, ents)
    
    print("✅ Processing complete!\n")

if __name__ == "__main__":
    main()