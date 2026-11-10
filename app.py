
import streamlit as st
import whisper
import tempfile
import os
import hashlib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LINGUA-AI",
    page_icon="🎙️",
    layout="wide"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0b1120, #111c35);
    color: #f1f5f9;
}

/* Main container */
.block-container {
    max-width: 1100px;
    padding: 2rem 2rem 3rem 2rem;
}

/* Main headings */
h1 {
    color: #60a5fa !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    text-align: center;
}

h2, h3 {
    color: #93c5fd !important;
    font-weight: 700 !important;
}

/* Text */
p, label {
    color: #dbeafe;
}

/* Header */
.header-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
}

/* Section headings */
.section-heading {
    color: #60a5fa;
    font-size: 21px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 13px 20px;
    font-size: 16px;
    font-weight: 700;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #4338ca);
    color: white;
    border: 1px solid #93c5fd;
    transform: translateY(-2px);
}

/* Audio uploader */
div[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.7);
    border: 1px dashed #60a5fa;
    border-radius: 15px;
    padding: 20px;
}

/* Radio buttons */
div[role="radiogroup"] {
    background: rgba(30, 41, 59, 0.5);
    padding: 15px;
    border-radius: 12px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #172554, #1e3a8a);
    border: 1px solid #3b82f6;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}

/* Result title */
.result-label {
    color: #bfdbfe;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 2px;
}

/* Identified language */
.result-language {
    color: #ffffff;
    font-size: 36px;
    font-weight: 800;
    margin: 12px 0;
}

/* Language icon */
.result-icon {
    font-size: 45px;
    margin-bottom: 10px;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(37, 99, 235, 0.15);
    border: 1px solid rgba(96, 165, 250, 0.4);
    padding: 18px;
    border-radius: 14px;
}

/* Audio player */
audio {
    width: 100%;
    border-radius: 12px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Progress */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 25px 10px;
    margin-top: 35px;
}

/* Divider */
hr {
    border-color: rgba(148, 163, 184, 0.2);
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


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.markdown("""
<div style="text-align:center; padding:15px 0 30px 0;">

    <div style="font-size:55px;">🎙️</div>

    <h1>LINGUA-AI</h1>

    <p class="header-subtitle">
        Multilingual Speech Recognition & Language Identification
    </p>

    <p style="color:#94a3b8; font-size:15px;">
        Identify the language spoken in your audio using Artificial Intelligence
    </p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. LANGUAGE INFORMATION
# ==========================================

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ur": "Urdu",
    "pa": "Punjabi"
}


# ==========================================
# 5. LOAD WHISPER MODEL
# ==========================================

@st.cache_resource
def load_model():
    return whisper.load_model("base", device="cpu")


with st.spinner("Loading AI language detection model..."):
    model = load_model()

st.success("Language detection model is ready!")


# ==========================================
# 6. SESSION STATE
# ==========================================

if "detected_language" not in st.session_state:
    st.session_state.detected_language = None

if "audio_id" not in st.session_state:
    st.session_state.audio_id = None

if "language_score" not in st.session_state:
    st.session_state.language_score = None


# ==========================================
# 7. AUDIO INPUT
# ==========================================

st.markdown(
    '<div class="section-heading">Step 1: Upload or Record Audio</div>',
    unsafe_allow_html=True
)

input_type = st.radio(
    "Select your audio input method:",
    ["Upload Audio", "Record Audio"],
    horizontal=True
)

if input_type == "Upload Audio":

    audio_file = st.file_uploader(
        "Upload your audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"]
    )

else:

    audio_file = st.audio_input(
        "Record your speech"
    )


# ==========================================
# 8. LANGUAGE IDENTIFICATION
# ==========================================

if audio_file is not None:

    audio_bytes = audio_file.getvalue()

    st.audio(audio_bytes)

    current_audio_id = hashlib.md5(
        audio_bytes
    ).hexdigest()

    if current_audio_id != st.session_state.audio_id:

        st.session_state.audio_id = current_audio_id
        st.session_state.detected_language = None
        st.session_state.language_score = None

    st.markdown(
        '<div class="section-heading">Step 2: Identify Spoken Language</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🌐 Identify Language",
        type="primary"
    ):

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

            with st.spinner(
                "Analysing audio and identifying language..."
            ):

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

                confidence_score = probabilities[detected_code]

                st.session_state.detected_language = detected_language
                st.session_state.language_score = confidence_score

        except Exception as error:

            st.error(
                f"Language identification failed: {error}"
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


    # ======================================
    # 9. DISPLAY IDENTIFIED LANGUAGE
    # ======================================

    if st.session_state.detected_language:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    IDENTIFIED LANGUAGE
                </div>

                <div class="result-icon">
                    🌐
                </div>

                <div class="result-language">
                    {st.session_state.detected_language}
                </div>

                <div style="color:#bfdbfe; font-size:14px;">
                    Language successfully identified
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.metric(
            "Model Prediction Score",
            f"{st.session_state.language_score * 100:.2f}%"
        )

        st.caption(
            "Prediction score is the model's estimate, "
            "not a verified accuracy percentage."
        )

else:

    st.info(
        "Please upload or record an audio file to begin."
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.markdown("""
<div class="footer">

    <hr>

    <p><b>LINGUA-AI</b></p>

    <p>
        AI-Powered Multilingual Language Identification
    </p>

    <p>
        Developed using Python | OpenAI Whisper | Streamlit
    </p>

    <p>
        Final Year Deep Learning Project | 2026
    </p>

</div>
""", unsafe_allow_html=True)
