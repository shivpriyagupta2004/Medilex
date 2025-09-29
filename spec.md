# AI Health Agent – Project Specification

## Scope
- **Languages:** Hindi & English
- **Connectivity model:** Online only
- **Doc types:** User-entered symptoms (text) and prescriptions (uploaded images)

## Goals
- Input: Symptoms text or prescription photo
- Output: Human-friendly explanation of condition, care steps, diet/rest suggestions, and red-flag alerts
- Translation into Hindi/English as needed

## Safety Rules
- Always show disclaimer: "This is informational only – not a diagnosis."
- Escalate for red flags: chest pain, severe breathlessness, heavy bleeding, high fever > 40°C, altered consciousness
- Always cite sources from WHO, MedlinePlus, or local Ministry of Health

## Tools
- LangChain for pipeline orchestration
- Chroma (local) vector DB for RAG
- Google Vision API for OCR
- MedCAT / scispaCy for clinical NER
- MedGPT / MedPaLM2 (if available) as LLM
- Google Translate API for Hindi

## Deliverables
- `ocr/` module for prescription text extraction
- `ner/` module for symptom & drug extraction
- `rag/` ingestion + retriever pipeline
- `app.py` (Streamlit UI)
- Logging & clinician review dashboard

## Next Steps
1. Collect 20 trusted health documents (WHO/MedlinePlus).
2. Build initial vector DB with Chroma.
3. Test OCR with 10 sample prescriptions.
4. Prototype LangChain RetrievalQA.
5. Add translation + Streamlit front-end.
