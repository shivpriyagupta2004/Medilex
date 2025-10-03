# ner.py
import re
from difflib import get_close_matches

# ---------- (optional) NLP - not used below but kept for compatibility ----------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_sci_sm")
    except Exception:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None
except Exception:
    nlp = None

# ---------- Canonical list ----------
CANON_MEDS = [
    "Betaloc", "Dorzolamide", "Cimetidine", "Oxprenolol",
    "Paracetamol", "Acetaminophen", "Ibuprofen", "Amoxicillin",
    "Aspirin", "Cetirizine", "Metformin", "Ciprofloxacin",
    "Azithromycin", "Omeprazole", "Atorvastatin", "Lisinopril",
    "Levothyroxine", "Metoprolol", "Amlodipine", "Simvastatin",
    "Losartan", "Gabapentin", "Hydrochlorothiazide", "Prednisone",
    "Montelukast", "Sertraline", "Furosemide", "Pantoprazole"
]
CANON_MEDS_LOWER = [m.lower() for m in CANON_MEDS]

# OCR aliases / misspellings
ALIASES = {
    "betsloe": "Betaloc",
    "beteloc": "Betaloc",
    "betaloc": "Betaloc",
    "vorzolaridum": "Dorzolamide",
    "dorzolamidum": "Dorzolamide",
    "oxprelel": "Oxprenolol",
    "oxprelol": "Oxprenolol",
}

# ---------- Regex maps ----------
FREQ_MAP = {
    "OD": "Once daily", "QD": "Once daily", "BD": "Twice daily",
    "BID": "Twice daily", "TID": "Three times daily",
    "QID": "Four times daily", "HS": "At bedtime",
    "QHS": "At bedtime", "PRN": "As needed", "SOS": "If necessary",
}

DOSE_UNITS = r"(?:mg|mcg|g|ml|iu|units|%)"
FREQ_RE = re.compile(r"\b(OD|QD|BD|BID|TID|QID|HS|QHS|PRN|SOS|Q\d+h)\b", re.I)
DOSE_RE = re.compile(rf"\b(\d{{1,4}}(?:\.\d{{1,2}})?)\s*({DOSE_UNITS})\b", re.I)
FORM_RE = re.compile(r"\b(tab|tablet|cap|capsule|syr(?:up)?|susp(?:ension)?|inj(?:ection)?|drops?)\b", re.I)

# ---------- Admin / Non-med filters ----------
ADMIN_KEYWORDS = [
    "dea", "lic", "medical centre", "medical center", "hospital",
    "name", "address", "age", "date", "signature", "sign", "doctor", "dr",
    "refill", "label", "stock", "presc", "usa", "new york", "street", "avenue",
    "road", "ny", "zip", "wtx", "adobe"
]

def _looks_like_admin_text(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    if any(k in low for k in ADMIN_KEYWORDS):
        return True
    # addressy lines: commas but no dose/freq/form
    if "," in low and not (DOSE_RE.search(low) or FREQ_RE.search(low) or FORM_RE.search(low)):
        return True
    # tiny tokens / RX marker
    if re.fullmatch(r"[rx\W\d]{1,4}", low):
        return True
    return False

# ---------- Normalization ----------
def _normalize_line(line: str) -> str:
    s = line
    s = (s.replace("\u2022", "-")  # •
           .replace("\u2014", "-") # —
           .replace("\u2013", "-"))# –
    s = s.replace("|", "1")
    # Fix OCR slips
    s = s.replace(" ag", " mg").replace(" m9", " mg").replace("m9", "mg")
    s = s.replace(" my", " mg").replace(" rng", " mg").replace(" mg.", " mg")
    s = re.sub(r"\b1\s*ab\b", "1 tab", s, flags=re.I)
    # digit/letter confusions
    s = re.sub(r"(?<=\d)[Oo](?=\d)", "0", s)
    s = re.sub(r"(?<=\d)[Il](?=\d)", "1", s)
    s = re.sub(r"(?<=\d)[Zz](?=\d)", "2", s)
    return re.sub(r"\s{2,}", " ", s).strip()

# ---------- Fuzzy canonicalization ----------
def _fuzzy_canon(name: str) -> str:
    q = name.lower().strip(" .:-")
    if q in ALIASES:
        return ALIASES[q]
    if q in CANON_MEDS_LOWER:
        return CANON_MEDS[CANON_MEDS_LOWER.index(q)]
    # prefer full-string high similarity
    match = get_close_matches(q, CANON_MEDS_LOWER, n=1, cutoff=0.6)
    if match:
        return CANON_MEDS[CANON_MEDS_LOWER.index(match[0])]
    # token-by-token fallback
    tokens = [t for t in re.split(r"[^\w]+", q) if len(t) > 2]
    for t in tokens:
        if t in ALIASES:
            return ALIASES[t]
        m = get_close_matches(t, CANON_MEDS_LOWER, n=1, cutoff=0.6)
        if m:
            return CANON_MEDS[CANON_MEDS_LOWER.index(m[0])]
    return name.title()

# ---------- Parser ----------
def _parse_line_to_med(line: str):
    s = _normalize_line(line)
    if _looks_like_admin_text(s):
        return None

    # must look like a med instruction
    freq_match = FREQ_RE.search(s)
    dose_match = DOSE_RE.search(s)
    has_form = FORM_RE.search(s)
    if not (freq_match or dose_match or has_form):
        return None

    # get dose/freq
    freq = freq_match.group(1).upper() if freq_match else ""
    dose = f"{dose_match.group(1)} {dose_match.group(2).lower()}" if dose_match else ""

    # drug name: use first word (robust) or left part before dash
    left = s.split("-")[0].strip()
    m = re.match(r"([A-Za-z][A-Za-z\-]{2,})", left)
    name_part = m.group(1) if m else left
    # remove trailing form/unit tokens
    name_part = re.sub(r"\b(tab|tablet|cap|capsule|mg|mcg|g|ml|iu|units|%)\b", "", name_part, flags=re.I).strip()

    if not name_part:
        return None

    canon = _fuzzy_canon(name_part)
    route = "Oral (Tablet)" if "tab" in s.lower() else ("Injection" if "inj" in s.lower() else "Oral")

    # final validation: avoid admin-like names
    if _looks_like_admin_text(canon.lower()):
        return None

    return {
        "name": canon,
        "dose": dose,
        "freq": freq,
        "freq_expanded": FREQ_MAP.get(freq, freq),
        "route": route,
        "raw": line
    }

# ---------- Extractor ----------
def _extract_meds(text: str):
    meds = []
    seen = set()
    for l in text.splitlines():
        if not l.strip():
            continue
        m = _parse_line_to_med(l)
        if not m:
            continue
        key = (m["name"].lower(), m["dose"], m["freq"])
        if key not in seen:
            seen.add(key)
            meds.append(m)
    return meds

# ---------- Public ----------
def extract_entities(text: str):
    return {
        "medications": _extract_meds(text),
        "symptoms": [],
        "diet": []
    }

# ---------- Test ----------
if __name__ == "__main__":
    demo = """MEDICAL CENTRE
824 14th Street
New York, NY 91743, USA
Name John Smith     Age 34
Address 162 Example St, NY   Date 09-11-12
RX
Betaloc 100mg - 1 tab BID
Dorzolamidum 10 mg - 1 tab BID
Cimetidine 50 mg - 2 tabs TID
Oxprelel 50m9 - 1 tab QD
Dr. Steve Johnson
REFILL 0 1 2 3 4 5 PRN"""
    result = extract_entities(demo)
    print("=== Test Results ===")
    for m in result["medications"]:
        print(m)
