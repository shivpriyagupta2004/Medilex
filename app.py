import streamlit as st
from PIL import Image
import pytesseract
import whisper
import pyttsx3
import tempfile
import os

from ner.ner import extract_entities
from rag.query import run_query


# ---- App Config ----
st.set_page_config(page_title="MediLex - Offline AI Healthcare", layout="centered")
st.title("🩺 MediLex - Offline Healthcare Assistant")
st.markdown("No APIs used. Runs **fully offline**.")


# ---- Input Method ----
option = st.radio("Choose input method:", ["Upload Prescription", "Type Symptoms", "Speak Symptoms"])
text = ""


# ---- Upload Prescription ----
if option == "Upload Prescription":
    uploaded_file = st.file_uploader("Upload a prescription image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Prescription", use_column_width=True)
        text = pytesseract.image_to_string(image, lang="eng")
        st.subheader("📄 OCR Extracted Text")
        st.text(text)


# ---- Type Symptoms ----
elif option == "Type Symptoms":
    text = st.text_area("Enter symptoms here:", "")


# ---- Speak Symptoms (Offline Whisper) ----
elif option == "Speak Symptoms":
    st.info("🎤 Speak symptoms (10s).")
    if st.button("Start Recording"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            os.system(f"arecord -d 10 -f cd {tmp_audio.name}")  # Linux example
            model = whisper.load_model("base")
            result = model.transcribe(tmp_audio.name)
            text = result["text"]
            st.success(f"You said: {text}")


# ---- MediLex Analysis ----
if st.button("Analyze with MediLex"):
    if not text.strip():
        st.warning("⚠️ Please provide input.")
    else:
        # --- NER ---
        entities = extract_entities(text)
        st.subheader("🔍 Detected Entities")
        st.json(entities)

        # --- RAG Explanation ---
        query = f"Explain in simple, patient-friendly terms: {text}"
        answer = run_query(query)

        # Add Diet & Rest Suggestions
        care_tips = """
        🥗 **Diet & Rest Suggestions**:
        - Stay hydrated
        - Eat light food (khichdi, soup, fruits)
        - Avoid oily/spicy food
        - Take proper rest
        - Consult doctor if symptoms worsen
        """
        final_answer = answer + "\n\n" + care_tips

        st.subheader("💡 Explanation (English)")
        st.write(final_answer)

        # --- Offline Text-to-Speech ---
        try:
            engine = pyttsx3.init()
            engine.save_to_file(final_answer, "output.mp3")
            engine.runAndWait()
            st.subheader("🔊 Listen")
            st.audio("output.mp3", format="audio/mp3")
        except Exception as e:
            st.error(f"TTS failed: {e}")
