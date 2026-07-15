"""
Meter Reading Extraction Pipeline
Connects a YOLO detector (finds the meter display) with TrOCR (reads the text)
Run with: streamlit run meter_reading_app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import re
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

# ---------------------------
# CONFIG - update these paths
# ---------------------------
YOLO_MODEL_PATH = "models/yolo_weights.pt"
TROCR_MODEL_PATH = "Sakshehehe/trocr-meter-final"  # loaded from Hugging Face Hub
CONFIDENCE_THRESHOLD = 0.4


# ---------------------------
# LOAD MODELS (cached so they only load once)
# ---------------------------
@st.cache_resource
def load_models():
    detector = YOLO(YOLO_MODEL_PATH)
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_PATH)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model.to(device)
    return detector, processor, trocr_model, device


detector, processor, trocr_model, device = load_models()


# ---------------------------
# STEP 1: Detect meter display region
# ---------------------------
def detect_meter(image: Image.Image):
    """
    Runs YOLO on the image, returns the cropped display region
    and the bounding box (for drawing/debugging).
    If multiple boxes are found, picks the one with highest confidence.
    """
    results = detector(image)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, None, 0.0

    # pick highest confidence box
    confidences = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confidences))
    best_conf = float(confidences[best_idx])

    if best_conf < CONFIDENCE_THRESHOLD:
        return None, None, best_conf

    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
    cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))

    return cropped, (int(x1), int(y1), int(x2), int(y2)), best_conf


# ---------------------------
# STEP 2: Preprocess cropped region before OCR
# ---------------------------
def preprocess_for_ocr(cropped_img: Image.Image):
    """Basic cleanup - convert to RGB, resize keeping aspect ratio, pad to square."""
    img = cropped_img.convert("RGB")

    target_size = 384
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h))

    # pad to square with white background
    padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(img, (paste_x, paste_y))

    return padded


# ---------------------------
# STEP 3: Run TrOCR
# ---------------------------
def extract_text(cropped_img: Image.Image):
    processed_img = preprocess_for_ocr(cropped_img)
    pixel_values = processor(images=processed_img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    generated_ids = trocr_model.generate(
        pixel_values,
        num_beams=5,          # beam search for cleaner output
        max_length=32
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


# ---------------------------
# STEP 4: Post-process output
# ---------------------------
def clean_reading(raw_text: str):
    """
    Light cleanup for meter readings that vary between numeric and
    alphanumeric (units like kWh, m3, etc). Fixes common OCR mix-ups
    and strips stray symbols, without being overly aggressive.
    """
    text = raw_text.strip()

    # common OCR character confusions in numeric context
    replacements = {
        "O": "0", "o": "0",
        "l": "1", "I": "1",
        "S": "5", "B": "8",
        "Z": "2",
    }

    # only apply digit-confusion fixes to characters surrounded by digits
    fixed_chars = list(text)
    for i, ch in enumerate(fixed_chars):
        if ch in replacements:
            left_digit = i > 0 and fixed_chars[i - 1].isdigit()
            right_digit = i < len(fixed_chars) - 1 and fixed_chars[i + 1].isdigit()
            if left_digit or right_digit:
                fixed_chars[i] = replacements[ch]
    text = "".join(fixed_chars)

    # remove stray double spaces / junk symbols but keep units and decimals
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z.\-/ ]", "", text)

    return text.strip()


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(
    page_title="Meter Reading Extraction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# CUSTOM CSS - professional, compact UI
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f4f6f8;
        color: #1f2933;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 1.6rem;
        max-width: 1200px;
    }

    .header-bar {
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        background: #1f2933;
        border: 1px solid #2c3644;
    }
    .header-bar h1 {
        font-size: 1.35rem;
        font-weight: 600;
        margin: 0;
        color: #ffffff;
        letter-spacing: -0.01em;
    }
    .header-bar p {
        color: #9aa5b1;
        margin: 0.15rem 0 0 0;
        font-size: 0.85rem;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e2e6ea;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.9rem;
    }
    .panel-label {
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .reading-panel {
        border-radius: 10px;
        padding: 1rem 1.1rem;
        border: 1px solid;
    }
    .reading-good {
        background: #f0faf4;
        border-color: #b7e4c7;
    }
    .reading-warn {
        background: #fff8ec;
        border-color: #f2d49b;
    }
    .reading-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: #1f2933;
    }
    .reading-status {
        font-size: 0.8rem;
        margin-top: 0.35rem;
        font-weight: 500;
    }
    .status-good { color: #1f8a4c; }
    .status-warn { color: #b7791f; }

    .conf-pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        background: #eef1f4;
        border: 1px solid #dde2e7;
        color: #475569;
        margin-left: 0.4rem;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 1.5px dashed #c3cad2;
        border-radius: 10px;
    }

    .stAlert { border-radius: 8px; }

    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e6ea;
        border-radius: 8px;
    }

    div[data-testid="stImage"] img {
        border-radius: 8px;
        max-height: 260px;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div class="header-bar">
    <h1>Meter Reading Extraction</h1>
    <p>Upload a meter image to detect the display region and extract the reading.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload meter image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Detecting meter display..."):
        cropped, box, confidence = detect_meter(image)

    if cropped is None:
        st.error(f"No meter display detected confidently (confidence: {confidence:.2f}). "
                  f"Try a clearer or closer image.")
    else:
        with st.spinner("Reading text..."):
            raw_text = extract_text(cropped)
            final_text = clean_reading(raw_text)

        with col2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel-label">Detected Display '
                f'<span class="conf-pill">conf {confidence:.2f}</span></div>',
                unsafe_allow_html=True,
            )
            st.image(cropped, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            is_low_conf = confidence < 0.6
            box_class = "reading-warn" if is_low_conf else "reading-good"
            status_class = "status-warn" if is_low_conf else "status-good"
            status_text = "Low detection confidence - please verify" if is_low_conf else "High confidence reading"

            st.markdown(f"""
            <div class="reading-panel {box_class}">
                <div class="panel-label">Reading</div>
                <div class="reading-value">{final_text}</div>
                <div class="reading-status {status_class}">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Debug info"):
            st.write(f"Raw TrOCR output: `{raw_text}`")
            st.write(f"Detection confidence: {confidence:.2f}")
            st.write(f"Bounding box: {box}")
