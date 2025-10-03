import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import tempfile
import os
import io

# Import custom modules
from ner.ner import extract_entities
from rag.query import run_query, search_medicine, search_symptom

# Page configuration
st.set_page_config(
    page_title="MediLex - Offline AI Healthcare",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🩺 MediLex</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Offline AI Healthcare Assistant - No APIs Required</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About MediLex")
    st.write("""
    MediLex is an **offline healthcare assistant** that helps you:
    - 📄 Extract information from prescriptions
    - 💊 Identify medicines and their usage
    - 🩺 Understand symptoms
    - 🥗 Get diet & lifestyle advice
    
    **Features:**
    - OCR for prescription reading
    - NER for entity extraction
    - RAG for medical knowledge
    - Voice input support
    - Text-to-Speech output
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "OCR Confidence",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Adjust OCR sensitivity"
    )
    
    show_raw_ocr = st.checkbox("Show Raw OCR Output", value=False)
    enable_tts = st.checkbox("Enable Text-to-Speech", value=True)

# Main content
st.divider()

# Input method selection
st.subheader("📝 Choose Input Method")
option = st.radio(
    "How would you like to provide information?",
    ["Upload Prescription", "Type Symptoms", "Voice Input"],
    horizontal=True
)

text = ""
uploaded_image = None

# Input handling
if option == "Upload Prescription":
    st.markdown('<div class="info-box">📸 Upload a clear image of your prescription</div>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            uploaded_image = image
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            # Preprocess image
            with st.spinner("🔄 Processing image..."):
                # Convert to grayscale
                gray = ImageOps.grayscale(image)
                # Auto contrast
                gray = ImageOps.autocontrast(gray, cutoff=2)
                # Resize if too small
                w, h = gray.size
                if min(w, h) < 900:
                    scale = 900 / min(w, h)
                    gray = gray.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                # Threshold
                gray = gray.point(lambda p: 255 if p > 160 else 0)
                # Sharpen
                gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=5))
            
            with col2:
                st.image(gray, caption="Processed Image", use_column_width=True)
            
            # Perform OCR
            with st.spinner("🔍 Extracting text from image..."):
                config = "--oem 3 --psm 6"
                text = pytesseract.image_to_string(gray, lang="eng", config=config)
            
            if show_raw_ocr and text:
                st.subheader("📄 Raw OCR Output")
                with st.expander("Click to view raw text"):
                    st.text_area("Extracted Text", text, height=200)
            
            if text.strip():
                st.success("✅ Text extracted successfully!")
            else:
                st.warning("⚠️ No text detected. Try a clearer image.")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

elif option == "Type Symptoms":
    st.markdown('<div class="info-box">✍️ Describe your symptoms in detail</div>', 
                unsafe_allow_html=True)
    
    text = st.text_area(
        "Enter your symptoms or medical query:",
        height=150,
        placeholder="Example: I have fever, cough, and headache for 3 days..."
    )
    
    # Common symptom quick buttons
    st.write("**Quick Symptoms:**")
    cols = st.columns(4)
    symptoms = ["Fever", "Cough", "Headache", "Nausea"]
    for i, symptom in enumerate(symptoms):
        if cols[i].button(symptom):
            text += f" {symptom.lower()}"
            st.rerun()

elif option == "Voice Input":
    st.markdown('<div class="info-box">🎤 Use voice to describe your symptoms</div>', 
                unsafe_allow_html=True)
    
    st.info("⚠️ Voice recording requires microphone permissions and additional setup.")
    
    if st.button("🎙️ Start Recording (10 seconds)"):
        try:
            import sounddevice as sd
            import soundfile as sf
            import whisper
            
            with st.spinner("🎤 Recording... Speak now!"):
                # Record audio
                duration = 10  # seconds
                fs = 16000  # Sample rate
                audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    sf.write(tmp_audio.name, audio, fs)
                    
                    # Transcribe with Whisper
                    with st.spinner("🔄 Converting speech to text..."):
                        model = whisper.load_model("base")
                        result = model.transcribe(tmp_audio.name)
                        text = result.get("text", "")
                    
                    # Clean up
                    os.unlink(tmp_audio.name)
            
            if text:
                st.success(f"✅ You said: {text}")
            else:
                st.warning("⚠️ Could not understand audio. Please try again.")
                
        except ImportError:
            st.error("❌ Voice input requires: pip install sounddevice soundfile openai-whisper")
        except Exception as e:
            st.error(f"❌ Recording error: {str(e)}")

# Analysis section
st.divider()

if st.button("🔬 Analyze with MediLex", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("⚠️ Please provide input first.")
    else:
        # Extract entities
        with st.spinner("🧠 Analyzing..."):
            entities = extract_entities(text)
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💊 Medicines", "🩺 Symptoms", "🥗 Diet", "💡 Explanation"])
        
        # Tab 1: Medicines
        with tab1:
            st.subheader("💊 Detected/Recommended Medicines")
            meds = entities.get("medications", [])
            
            if meds:
                # Create formatted table
                for i, med in enumerate(meds, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{i}. {med.get('name', 'Unknown')}**")
                            if med.get('dose'):
                                st.write(f"📏 **Dose:** {med['dose']}")
                            if med.get('freq_expanded'):
                                st.write(f"⏰ **Frequency:** {med['freq_expanded']}")
                            if med.get('route'):
                                st.write(f"💉 **Route:** {med['route']}")
                        
                        with col2:
                            if st.button(f"ℹ️ Info", key=f"med_{i}"):
                                info = search_medicine(med.get('name', ''))
                                st.info(info)
                        
                        st.divider()
                
                # Download option
                med_text = "\n".join([
                    f"{m['name']} - {m.get('dose', 'N/A')} - {m.get('freq_expanded', 'N/A')}"
                    for m in meds
                ])
                st.download_button(
                    label="📥 Download Medicine List",
                    data=med_text,
                    file_name="medicines.txt",
                    mime="text/plain"
                )
            else:
                st.info("ℹ️ No medicines detected. Try uploading a clearer prescription or typing medicine names.")
        
        # Tab 2: Symptoms
        with tab2:
            st.subheader("🩺 Identified Symptoms")
            symptoms = entities.get("symptoms", [])
            
            if symptoms:
                for symptom in symptoms:
                    with st.expander(f"🔍 {symptom}"):
                        info = search_symptom(symptom)
                        st.write(info)
            else:
                st.info("ℹ️ No specific symptoms identified. Describe your condition for better results.")
        
        # Tab 3: Diet & Lifestyle
        with tab3:
            st.subheader("🥗 Diet & Lifestyle Recommendations")
            diet = entities.get("diet", [])
            
            if diet:
                for i, item in enumerate(diet, 1):
                    st.markdown(f"{i}. {item}")
            else:
                st.markdown("""
                **General Health Tips:**
                1. Stay well-hydrated (8-10 glasses of water daily)
                2. Eat light, nutritious meals
                3. Avoid oily and spicy foods
                4. Get adequate rest (7-8 hours sleep)
                5. Maintain good hygiene
                """)
            
            st.markdown('<div class="warning-box">⚠️ <strong>Important:</strong> These are general guidelines. Always follow your doctor\'s specific advice.</div>', 
                        unsafe_allow_html=True)
        
        # Tab 4: Explanation
        with tab4:
            st.subheader("💡 Medical Knowledge Base Explanation")
            
            # Generate query based on input
            if meds:
                query = f"Explain these medicines and their uses: {', '.join([m['name'] for m in meds])}"
            elif symptoms:
                query = f"Explain symptoms and treatment: {', '.join(symptoms)}"
            else:
                query = f"Explain: {text[:200]}"
            
            with st.spinner("🔍 Searching knowledge base..."):
                answer = run_query(query, top_k=3)
            
            st.markdown(answer)
            
            # Add disclaimer
            st.markdown('<div class="warning-box">🏥 <strong>Medical Disclaimer:</strong> This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.</div>', 
                        unsafe_allow_html=True)
            
            # Text-to-Speech
            if enable_tts:
                st.divider()
                st.subheader("🔊 Listen to Explanation")
                
                if st.button("🎧 Generate Audio"):
                    try:
                        import pyttsx3
                        
                        with st.spinner("🔊 Generating audio..."):
                            engine = pyttsx3.init()
                            
                            # Configure voice
                            engine.setProperty('rate', 150)  # Speed
                            engine.setProperty('volume', 0.9)  # Volume
                            
                            # Save to file
                            audio_file = "output.mp3"
                            engine.save_to_file(answer, audio_file)
                            engine.runAndWait()
                        
                        # Play audio
                        if os.path.exists(audio_file):
                            with open(audio_file, "rb") as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format="audio/mp3")
                            os.remove(audio_file)
                        else:
                            st.error("❌ Audio generation failed.")
                            
                    except Exception as e:
                        st.error(f"❌ TTS Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>MediLex v1.0</strong> - Offline Healthcare Assistant</p>
    <p>🔒 Your data stays on your device | 🌐 Works completely offline</p>
    <p style='font-size: 0.8rem;'>⚠️ Not a substitute for professional medical advice</p>
</div>
""", unsafe_allow_html=True)