
import streamlit as st
import whisper
import tempfile
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LINGUA-AI",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ LINGUA-AI")
st.subheader("Multilingual Speech Recognition System")

st.write(
    "Upload or record audio, identify the spoken language "
    "and convert speech into text."
)

# ==========================================
# 2. LANGUAGE INFORMATION
# ==========================================

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu"
}

# ==========================================
# 3. LOAD WHISPER MODEL
# ==========================================

@st.cache_resource
def load_model():
    return whisper.load_model("base", device="cpu")


with st.spinner("Loading speech recognition model..."):
    model = load_model()

st.success("Model ready!")

# ==========================================
# 4. SESSION STATE
# ==========================================

if "transcription" not in st.session_state:
    st.session_state.transcription = None

if "detected_language" not in st.session_state:
    st.session_state.detected_language = None

if "audio_id" not in st.session_state:
    st.session_state.audio_id = None

# ==========================================
# 5. AUDIO INPUT
# ==========================================

st.header("Step 1: Upload or Record Audio")

input_type = st.radio(
    "Select audio input method:",
    ["Upload Audio", "Record Audio"],
    horizontal=True
)

if input_type == "Upload Audio":

    audio_file = st.file_uploader(
        "Upload your audio",
        type=["wav", "mp3", "m4a", "ogg", "flac"]
    )

else:

    audio_file = st.audio_input("Record your speech")

# ==========================================
# 6. AUDIO PROCESSING
# ==========================================

if audio_file is not None:

    audio_bytes = audio_file.getvalue()

    st.audio(audio_bytes)

    # Reset previous result when audio changes
    import hashlib

    current_audio_id = hashlib.md5(audio_bytes).hexdigest()

    if current_audio_id != st.session_state.audio_id:

        st.session_state.audio_id = current_audio_id
        st.session_state.transcription = None
        st.session_state.detected_language = None

    # ======================================
    # 7. TRANSCRIBE BUTTON
    # ======================================

    if st.button("🎙️ Transcribe Audio", type="primary"):

        temp_path = None

        try:

            suffix = (
                os.path.splitext(audio_file.name)[1]
                if hasattr(audio_file, "name")
                else ".wav"
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            # Automatic language detection
            with st.spinner("Identifying language..."):

                audio = whisper.load_audio(temp_path)

                audio = whisper.pad_or_trim(audio)

                mel = whisper.log_mel_spectrogram(
                    audio,
                    n_mels=model.dims.n_mels
                ).to(model.device)

                _, probabilities = model.detect_language(mel)

                detected_code = max(
                    probabilities,
                    key=probabilities.get
                )

                detected_language = LANGUAGE_NAMES.get(
                    detected_code,
                    detected_code.upper()
                )

            # Speech transcription
            with st.spinner("Transcribing your audio..."):

                result = model.transcribe(
                    temp_path,
                    language=detected_code,
                    task="transcribe",
                    fp16=False,
                    temperature=0
                )

                recognized_text = result["text"].strip()

            # Save results
            st.session_state.detected_language = detected_language
            st.session_state.transcription = recognized_text

        except Exception as error:

            st.error(f"Error: {error}")

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # ======================================
    # 8. DISPLAY TRANSCRIPTION
    # ======================================

    if st.session_state.transcription is not None:

        st.divider()

        st.success("Transcription completed successfully!")

        st.subheader("Transcription Result")

        st.write(
            f"**Detected Language:** "
            f"{st.session_state.detected_language}"
        )

        st.text_area(
            "Your Transcribed Text",
            value=st.session_state.transcription,
            height=220
        )

        # ==================================
        # 9. DOWNLOAD BUTTON
        # ==================================

        st.download_button(
            label="⬇️ Download Transcription",
            data=st.session_state.transcription,
            file_name="LINGUA_AI_Transcription.txt",
            mime="text/plain"
        )

else:

    st.info("Upload or record audio to start.")

# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "LINGUA-AI | Multilingual Speech Recognition "
    "using OpenAI Whisper"
)

# ==========================================
# CUSTOM CSS - LINGUA-AI
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(135deg, #0b1120, #111c35);
    color: #f1f5f9;
}

/* Main content width */
.block-container {
    max-width: 1100px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Main title */
h1 {
    color: #60a5fa !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    text-align: center;
    letter-spacing: 1px;
}

/* Subheadings */
h2, h3 {
    color: #93c5fd !important;
    font-weight: 700 !important;
}

/* Paragraphs */
p, label, .stMarkdown {
    color: #dbeafe;
}

/* Header subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 35px;
}

/* Section containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 16px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 700;
    transition: 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #4338ca);
    border: 1px solid #93c5fd;
    transform: translateY(-2px);
}

/* Download button */
.stDownloadButton > button {
    width: 100%;
    background: #059669;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 700;
}

.stDownloadButton > button:hover {
    background: #047857;
    color: white;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.8);
    border: 1px dashed #60a5fa;
    border-radius: 14px;
    padding: 20px;
}

/* Audio player */
audio {
    width: 100%;
    border-radius: 10px;
}

/* Text area */
.stTextArea textarea {
    background: #111827 !important;
    color: #f8fafc !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}

/* Radio buttons */
.stRadio {
    background: rgba(30, 41, 59, 0.6);
    padding: 12px;
    border-radius: 12px;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(37, 99, 235, 0.12);
    border: 1px solid rgba(96, 165, 250, 0.3);
    padding: 18px;
    border-radius: 14px;
}

/* Success message */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: rgba(148, 163, 184, 0.2);
    margin-top: 25px;
    margin-bottom: 25px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 20px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)
