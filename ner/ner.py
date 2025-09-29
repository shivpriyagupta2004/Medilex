# ner/ner.py
try:
    import spacy
    try:
        nlp = spacy.load("en_core_sci_sm")  # if you installed scispacy
    except Exception:
        nlp = spacy.load("en_core_web_sm")  # fallback to standard spacy model
except Exception:
    nlp = None

# Basic lists for fallback
MED_LIST = [
    "paracetamol", "acetaminophen", "ibuprofen", "amoxicillin",
    "aspirin", "cetirizine", "metformin", "ciprofloxacin"
]

SYMPTOM_KEYWORDS = [
    "fever", "cough", "sore throat", "headache", "breath", "breathing",
    "nausea", "vomit", "pain", "dizziness", "fatigue"
]

def extract_entities(text: str):
    text_lower = text.lower()
    meds = [m for m in MED_LIST if m in text_lower]

    symptoms = set()
    if nlp:
        doc = nlp(text)
        for ent in getattr(doc, "ents", []):
            ent_text = ent.text.strip()
            if any(k in ent_text.lower() for k in SYMPTOM_KEYWORDS):
                symptoms.add(ent_text)
        for nc in getattr(doc, "noun_chunks", []):
            nc_text = nc.text.strip()
            if any(k in nc_text.lower() for k in SYMPTOM_KEYWORDS):
                symptoms.add(nc_text)
    else:
        for kw in SYMPTOM_KEYWORDS:
            if kw in text_lower:
                symptoms.add(kw)

    return {"medications": meds, "symptoms": sorted(symptoms)}

if __name__ == "__main__":
    print(extract_entities("High fever 39C, sore throat. Taking paracetamol 500mg."))
