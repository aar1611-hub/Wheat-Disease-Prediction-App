import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="WheatSense AI | Wheat Health Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "wheat_mobilenetv2.keras"
CLASS_PATH = "mobilenet_class_names.json"
IMAGE_SIZE = (160, 160)

# --------------------------------------------------
# CUSTOM CSS  (refined, animated styling)
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap');

    /* ---------- DESIGN TOKENS ---------- */
    :root {
        --ink: #10281C;
        --forest: #14412B;
        --forest-deep: #0B2417;
        --leaf: #2F8A57;
        --leaf-dark: #1F6B40;
        --grain: #E0A93B;
        --grain-soft: #F7EBC9;
        --mist: #F1F4EC;
        --card: #FFFFFF;
        --line: #DFE6D8;
        --muted: #667A6C;
        --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
        --ear: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='260' viewBox='0 0 120 260'%3E%3Cpath d='M60 260V30' stroke='white' stroke-width='2.5' stroke-linecap='round' fill='none'/%3E%3Cg fill='white'%3E%3Cellipse cx='60' cy='20' rx='6' ry='16'/%3E%3Cellipse cx='45' cy='60' rx='7' ry='16' transform='rotate(-28 45 60)'/%3E%3Cellipse cx='75' cy='60' rx='7' ry='16' transform='rotate(28 75 60)'/%3E%3Cellipse cx='45' cy='100' rx='7' ry='16' transform='rotate(-28 45 100)'/%3E%3Cellipse cx='75' cy='100' rx='7' ry='16' transform='rotate(28 75 100)'/%3E%3Cellipse cx='45' cy='140' rx='7' ry='16' transform='rotate(-28 45 140)'/%3E%3Cellipse cx='75' cy='140' rx='7' ry='16' transform='rotate(28 75 140)'/%3E%3Cellipse cx='45' cy='180' rx='7' ry='16' transform='rotate(-28 45 180)'/%3E%3Cellipse cx='75' cy='180' rx='7' ry='16' transform='rotate(28 75 180)'/%3E%3C/g%3E%3C/svg%3E");
    }

    /* ---------- KEYFRAMES ---------- */
    @keyframes heroShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes sway {
        from { transform: rotate(-1.5deg); }
        to   { transform: rotate(2deg); }
    }

    @keyframes float {
        from { transform: translate(0, 0) scale(1); }
        to   { transform: translate(-30px, 22px) scale(1.12); }
    }

    @keyframes drift {
        0%   { transform: translate3d(0, 0, 0) scale(1); }
        50%  { transform: translate3d(3%, -4%, 0) scale(1.08); }
        100% { transform: translate3d(-3%, 3%, 0) scale(1); }
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-22px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    @keyframes pop {
        0%   { opacity: 0; transform: scale(0.82); }
        60%  { opacity: 1; transform: scale(1.06); }
        100% { opacity: 1; transform: scale(1); }
    }

    @keyframes shine {
        0%   { background-position: 130% 0; }
        100% { background-position: -130% 0; }
    }

    @keyframes flow {
        from { background-position: 0% 50%; }
        to   { background-position: 200% 50%; }
    }

    @keyframes pollen {
        0%   { transform: translate3d(0, 0, 0) scale(0.6); opacity: 0; }
        12%  { opacity: 0.85; }
        85%  { opacity: 0.5; }
        100% { transform: translate3d(34px, -380px, 0) scale(1.1); opacity: 0; }
    }

    @keyframes breathe {
        0%   { box-shadow: 0 10px 22px -10px rgba(31, 107, 64, 0.65), 0 0 0 0 rgba(47, 138, 87, 0.45); }
        70%  { box-shadow: 0 10px 22px -10px rgba(31, 107, 64, 0.65), 0 0 0 16px rgba(47, 138, 87, 0); }
        100% { box-shadow: 0 10px 22px -10px rgba(31, 107, 64, 0.65), 0 0 0 0 rgba(47, 138, 87, 0); }
    }

    @keyframes dropzone {
        0%, 100% { box-shadow: 0 0 0 0 rgba(47, 138, 87, 0); border-color: #9DBBA3; }
        50%      { box-shadow: 0 0 0 9px rgba(47, 138, 87, 0.09); border-color: #4E9E6F; }
    }

    @keyframes grow {
        from { transform: scaleX(0); }
        to   { transform: scaleX(1); }
    }

    @keyframes imgIn {
        from { opacity: 0; filter: blur(14px); transform: scale(0.96); }
        to   { opacity: 1; filter: blur(0); transform: scale(1); }
    }

    @keyframes pulseText {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.45; }
    }

    /* ---------- GLOBAL ---------- */
    .stApp {
        background:
            radial-gradient(900px 480px at 100% -5%, rgba(224, 169, 59, 0.13), transparent 60%),
            radial-gradient(800px 520px at -10% 105%, rgba(47, 138, 87, 0.12), transparent 60%),
            var(--mist);
        color: var(--ink);
        font-family: 'Figtree', sans-serif;
    }

    /* slow-moving colour blobs behind the whole page */
    .stApp::before {
        content: "";
        position: fixed;
        inset: -10%;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(420px 420px at 15% 25%, rgba(47, 138, 87, 0.10), transparent 70%),
            radial-gradient(380px 380px at 85% 70%, rgba(224, 169, 59, 0.12), transparent 70%),
            radial-gradient(300px 300px at 55% 10%, rgba(109, 190, 138, 0.10), transparent 70%);
        animation: drift 22s ease-in-out infinite alternate;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3 {
        font-family: 'Bricolage Grotesque', sans-serif !important;
        color: var(--forest) !important;
        letter-spacing: -0.02em;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: var(--forest);
        font-weight: 700;
    }

    :focus-visible {
        outline: 3px solid rgba(224, 169, 59, 0.75) !important;
        outline-offset: 2px;
    }

    /* ---------- HERO ---------- */
    .hero {
        background: linear-gradient(120deg, #0B2417 0%, #14412B 40%, #1F6B40 70%, #14412B 100%);
        background-size: 220% 220%;
        animation: heroShift 16s ease-in-out infinite, rise 0.9s var(--ease-out) both;
        padding: 3rem 3rem 3.1rem 3rem;
        border-radius: 28px;
        color: #FFFFFF;
        margin-bottom: 2.2rem;
        box-shadow:
            0 24px 48px -24px rgba(11, 36, 23, 0.55),
            inset 0 0 0 1px rgba(255, 255, 255, 0.07);
        position: relative;
        overflow: hidden;
        isolation: isolate;
    }

    /* wheat ears, drawn in the background */
    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: -1;
        background-image: var(--ear), var(--ear), var(--ear);
        background-repeat: no-repeat;
        background-position: right 4% bottom -50px, right 17% bottom -95px, right 29% bottom -130px;
        background-size: 190px, 140px, 105px;
        opacity: 0.11;
        transform-origin: bottom right;
        animation: sway 9s ease-in-out infinite alternate;
    }

    /* warm sun glow */
    .hero::after {
        content: "";
        position: absolute;
        right: -90px;
        top: -110px;
        width: 340px;
        height: 340px;
        z-index: -1;
        background: radial-gradient(circle, rgba(224, 169, 59, 0.38) 0%, transparent 68%);
        border-radius: 50%;
        animation: float 10s ease-in-out infinite alternate;
    }

    /* floating pollen / grain particles */
    .grain {
        position: absolute;
        bottom: -14px;
        left: var(--x, 50%);
        width: var(--s, 6px);
        height: var(--s, 6px);
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #FFF3CC, #E0A93B 70%);
        box-shadow: 0 0 10px rgba(224, 169, 59, 0.7);
        opacity: 0;
        z-index: 0;
        pointer-events: none;
        animation: pollen var(--t, 9s) linear var(--d, 0s) infinite;
    }

    .hero .eyebrow,
    .hero h1,
    .hero p {
        position: relative;
        z-index: 1;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 3.1rem;
        margin: 0;
        letter-spacing: -0.035em;
        font-weight: 800;
        line-height: 1.08;
        padding: 0;
        background: linear-gradient(100deg, #FFFFFF 0%, #FFFFFF 38%, #F8DE98 50%, #FFFFFF 62%, #FFFFFF 100%);
        background-size: 260% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rise 0.8s 0.25s var(--ease-out) both, shine 6s 1.2s ease-in-out infinite;
    }

    .hero p {
        color: #D7EADD;
        font-size: 1.05rem;
        margin: 0.9rem 0 0 0;
        max-width: 580px;
        line-height: 1.65;
        animation: rise 0.8s 0.4s var(--ease-out) both;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #F7EBC9;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(247, 235, 201, 0.28);
        backdrop-filter: blur(6px);
        padding: 0.38rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 1.1rem;
        animation: rise 0.8s 0.1s var(--ease-out) both;
    }

    /* ---------- SECTION LABEL ---------- */
    .section-label {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--forest);
        margin-bottom: 1rem;
        animation: slideIn 0.7s 0.5s var(--ease-out) both;
    }

    .section-label::before {
        content: "";
        width: 26px;
        height: 4px;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--grain), var(--leaf), var(--grain));
        background-size: 200% 100%;
        animation: flow 3s linear infinite;
    }

    /* ---------- PANELS ---------- */
    .panel {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.75rem;
        min-height: 150px;
        box-shadow: 0 1px 2px rgba(16, 40, 28, 0.04), 0 10px 24px -18px rgba(16, 40, 28, 0.25);
        position: relative;
        overflow: hidden;
        animation: rise 0.7s 0.15s var(--ease-out) both;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
    }

    .panel::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: linear-gradient(180deg, var(--grain), var(--leaf), var(--grain));
        background-size: 100% 300%;
        animation: panelFlow 5s linear infinite;
    }

    @keyframes panelFlow {
        from { background-position: 0% 0%; }
        to   { background-position: 0% 300%; }
    }

    .panel:hover {
        border-color: #C4D4C3;
        box-shadow: 0 1px 2px rgba(16, 40, 28, 0.04), 0 18px 34px -18px rgba(16, 40, 28, 0.3);
        transform: translateY(-3px);
    }

    .panel-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--forest);
        margin-bottom: 0.4rem;
        letter-spacing: -0.015em;
    }

    .muted {
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.65;
    }

    /* ---------- RESULT ---------- */
    .result-label {
        display: inline-block;
        color: var(--leaf-dark);
        background: #E4F1E8;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.4rem 0 0.7rem 0;
        animation: slideIn 0.6s var(--ease-out) both;
    }

    .result-name {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--leaf-dark);
        line-height: 1.25;
        overflow-wrap: anywhere;
        letter-spacing: -0.025em;
        transform-origin: left center;
        animation: pop 0.7s 0.1s var(--ease-out) both;
    }

    .score-number {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 3.4rem;
        font-weight: 800;
        margin-top: 0.3rem;
        line-height: 1.1;
        letter-spacing: -0.04em;
        background: linear-gradient(90deg, #14412B, #2F8A57, #E0A93B, #2F8A57, #14412B);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        display: inline-block;
        transform-origin: left center;
        animation: pop 0.8s 0.25s var(--ease-out) both, flow 5s linear infinite;
    }

    /* ---------- BUTTONS ---------- */
    div.stButton > button {
        background: linear-gradient(135deg, #1F6B40 0%, #2F8A57 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 14px;
        padding: 0.9rem 1.4rem;
        font-family: 'Figtree', sans-serif;
        font-weight: 700;
        font-size: 1.02rem;
        width: 100%;
        letter-spacing: 0.2px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 22px -10px rgba(31, 107, 64, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        animation: breathe 2.8s ease-out infinite;
        transition: transform 0.2s ease, filter 0.25s ease;
    }

    /* light sweep on hover */
    div.stButton > button::after {
        content: "";
        position: absolute;
        top: 0;
        left: -80%;
        width: 50%;
        height: 100%;
        background: linear-gradient(100deg, transparent, rgba(255, 255, 255, 0.35), transparent);
        transform: skewX(-18deg);
        transition: left 0.6s ease;
    }

    div.stButton > button:hover {
        color: #FFFFFF;
        border: none;
        transform: translateY(-3px);
        filter: brightness(1.07);
        animation: none;
        box-shadow: 0 18px 30px -12px rgba(31, 107, 64, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.25);
    }

    div.stButton > button:hover::after {
        left: 130%;
    }

    div.stButton > button:active {
        transform: scale(0.97);
    }

    div.stButton > button:focus:not(:active) {
        color: #FFFFFF;
        border: none;
    }

    /* ---------- FILE UPLOADER ---------- */
    [data-testid="stFileUploader"] {
        background: var(--card);
        border: 1.5px dashed #9DBBA3;
        border-radius: 20px;
        padding: 1rem 1.1rem;
        animation: dropzone 3.6s ease-in-out infinite;
        transition: background 0.25s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--leaf);
        background: #FBFDF8;
        animation-play-state: paused;
        box-shadow: 0 0 0 6px rgba(47, 138, 87, 0.12);
    }

    [data-testid="stFileUploader"] label {
        font-weight: 700 !important;
        color: var(--forest) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }

    [data-testid="stFileUploader"] section button {
        border-radius: 10px;
        border: 1.5px solid var(--leaf);
        color: var(--leaf-dark);
        font-weight: 600;
        background: #FFFFFF;
        transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
    }

    [data-testid="stFileUploader"] section button:hover {
        background: var(--leaf);
        color: #FFFFFF;
        border-color: var(--leaf);
        transform: translateY(-1px);
    }

    /* ---------- PROGRESS BARS ---------- */
    div[data-testid="stProgress"] [role="progressbar"],
    div[data-testid="stProgressBar"] > div {
        background-color: #E6EDE2;
        border-radius: 20px;
        height: 12px;
        overflow: hidden;
    }

    div[data-testid="stProgress"] [role="progressbar"] > div,
    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #2F8A57, #6DBE8A, #E0A93B, #6DBE8A, #2F8A57);
        background-size: 200% 100%;
        border-radius: 20px;
        height: 12px;
        transform-origin: left center;
        transition: width 0.9s var(--ease-out);
        animation: grow 1.1s var(--ease-out) both, flow 4s linear infinite;
    }

    /* ---------- INFO NOTE ---------- */
    .info-note {
        background: linear-gradient(100deg, #FBF3DC 0%, #F1F6EC 55%);
        border: 1px solid #EADDB5;
        border-left: 4px solid var(--grain);
        border-radius: 12px;
        padding: 1.05rem 1.3rem;
        color: #3C5546;
        font-size: 0.9rem;
        line-height: 1.65;
        margin-top: 1.25rem;
        box-shadow: 0 1px 2px rgba(16, 40, 28, 0.03);
        animation: slideIn 0.7s 0.2s var(--ease-out) both;
    }

    .info-note b {
        color: var(--forest);
    }

    /* ---------- DIVIDER ---------- */
    hr {
        border: none;
        border-top: 1px solid var(--line);
        margin: 1.4rem 0;
    }

    /* ---------- ALERTS ---------- */
    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid var(--line);
        animation: slideIn 0.6s var(--ease-out) both;
    }

    /* ---------- CAPTIONS ---------- */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        font-size: 0.83rem !important;
    }

    /* ---------- IMAGE ---------- */
    [data-testid="stImage"] img {
        border-radius: 18px;
        border: 1px solid var(--line);
        box-shadow: 0 14px 30px -18px rgba(16, 40, 28, 0.4);
        animation: imgIn 0.9s var(--ease-out) both;
        transition: transform 0.35s ease, box-shadow 0.35s ease;
    }

    [data-testid="stImage"] img:hover {
        transform: scale(1.015);
        box-shadow: 0 22px 40px -18px rgba(16, 40, 28, 0.45);
    }

    /* ---------- FOOTER ---------- */
    .footer {
        border-top: 1px solid var(--line);
        margin-top: 3.2rem;
        padding-top: 1.6rem;
        color: var(--muted);
        text-align: center;
        font-size: 0.85rem;
        letter-spacing: 0.2px;
        line-height: 1.75;
        animation: rise 0.9s 0.6s var(--ease-out) both;
    }

    .footer b {
        color: var(--forest);
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 0.98rem;
    }

    /* ---------- SPINNER ---------- */
    .stSpinner > div {
        border-top-color: var(--leaf-dark) !important;
    }

    .stSpinner {
        animation: pulseText 1.4s ease-in-out infinite;
    }

    /* ---------- RESPONSIVE ---------- */
    @media (max-width: 700px) {
        .hero {
            padding: 1.9rem 1.5rem;
            border-radius: 20px;
        }

        .hero h1 {
            font-size: 2.1rem;
        }

        .hero p {
            font-size: 0.97rem;
        }

        .panel {
            padding: 1.25rem;
        }

        .result-name {
            font-size: 1.65rem;
        }

        .score-number {
            font-size: 2.6rem;
        }
    }

    /* ---------- REDUCED MOTION ---------- */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL AND CLASS LABELS
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <span class="grain" aria-hidden="true" style="--x:6%;  --s:5px; --t:9s;  --d:0s"></span>
    <span class="grain" aria-hidden="true" style="--x:15%; --s:7px; --t:12s; --d:2.5s"></span>
    <span class="grain" aria-hidden="true" style="--x:26%; --s:4px; --t:8s;  --d:5s"></span>
    <span class="grain" aria-hidden="true" style="--x:38%; --s:6px; --t:11s; --d:1s"></span>
    <span class="grain" aria-hidden="true" style="--x:51%; --s:5px; --t:10s; --d:3.5s"></span>
    <span class="grain" aria-hidden="true" style="--x:63%; --s:8px; --t:13s; --d:0.5s"></span>
    <span class="grain" aria-hidden="true" style="--x:74%; --s:4px; --t:9s;  --d:4s"></span>
    <span class="grain" aria-hidden="true" style="--x:84%; --s:6px; --t:11s; --d:6s"></span>
    <span class="grain" aria-hidden="true" style="--x:93%; --s:5px; --t:8s;  --d:2s"></span>
    <div class="eyebrow">🌾 Intelligent Agriculture</div>
    <h1>WheatSense AI</h1>
    <p>
        Deep learning for preliminary wheat image classification.
        Upload an image to explore its predicted health category.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">Wheat image analysis</div>',
    unsafe_allow_html=True
)

# Load resources with a friendly error message
try:
    model = load_model()
    class_names = load_classes()

    if len(class_names) != model.output_shape[-1]:
        st.error(
            "The class-label count does not match the model output. "
            "Please check the model and class-label files."
        )
        st.stop()

except Exception as error:
    st.error(
        "Unable to load the model or class labels. "
        "Check that the required files are in the project folder."
    )
    st.code(str(error))
    st.stop()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a wheat image",
    type=["jpg", "jpeg", "png", "webp","avif"],
    help="Choose a clear wheat image in JPG, PNG, or WebP format."
)

if uploaded_file is None:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Welcome to WheatSense AI</div>
        <p class="muted">
            Your uploaded image and prediction results will appear
            side by side here. Start by selecting a wheat image above.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-note">
        <b>About this tool:</b> This educational prototype predicts
        one of two image classes: Healthy Wheat or Sick Wheat.
        It does not provide a confirmed agricultural diagnosis.
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Prepare the image exactly as required by MobileNetV2
        resized_image = image.resize(IMAGE_SIZE)
        image_array = np.array(resized_image, dtype=np.float32)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        left_col, right_col = st.columns(
            [1, 1],
            gap="large",
            vertical_alignment="top"
        )

        with left_col:
            st.markdown("""
            <div class="panel">
                <div class="panel-title">Uploaded image</div>
                <p class="muted">Preview of the selected wheat image.</p>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                image,
                width="stretch",
                caption="Input image"
            )

            st.caption(
                f"Image dimensions: {image.width} × {image.height} pixels"
            )

        with right_col:
            st.markdown("""
            <div class="panel">
                <div class="panel-title">Prediction results</div>
                <p class="muted">
                    Run the model to see the predicted category
                    and class probability scores.
                </p>
            </div>
            """, unsafe_allow_html=True)

            predict_clicked = st.button(
                "✦  Analyse wheat image",
                type="primary",
                use_container_width=True
            )

            if predict_clicked:
                with st.spinner("Analysing image with the AI model..."):
                    raw_output = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                probabilities = np.asarray(
                    raw_output, dtype=np.float64
                ).reshape(-1)

                if (
                    len(probabilities) != len(class_names)
                    or not np.all(np.isfinite(probabilities))
                    or np.any(probabilities < 0)
                    or not np.isclose(probabilities.sum(), 1.0, atol=0.02)
                ):
                    st.error(
                        "The model returned unexpected scores. "
                        "Please check the model output and preprocessing."
                    )
                else:
                    predicted_index = int(np.argmax(probabilities))
                    predicted_class = class_names[predicted_index]
                    confidence = float(
                        probabilities[predicted_index] * 100
                    )

                    # Store the latest result across Streamlit reruns
                    st.session_state["wheat_result"] = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "probabilities": probabilities.tolist(),
                        "filename": uploaded_file.name
                    }

            result = st.session_state.get("wheat_result")

            # Only display a result for the currently selected image
            if (
                result is not None
                and result["filename"] == uploaded_file.name
            ):
                st.markdown(
                    '<div class="result-label">Predicted category</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-name">{result["class"]}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="score-number">'
                    f'{result["confidence"]:.2f}%</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Model score for the predicted category; "
                    "not a guarantee of real-world correctness."
                )

                st.divider()

                st.markdown("**Class probability scores**")

                for name, probability in zip(
                    class_names, result["probabilities"]
                ):
                    percentage = probability * 100

                    st.markdown(
                        f"**{name}** — {percentage:.2f}%"
                    )

                    st.progress(
                        min(1.0, max(0.0, float(probability)))
                    )

                st.markdown("""
                <div class="info-note">
                    <b>Important:</b> This is an experimental
                    image-classification result, not a confirmed
                    plant diagnosis. Consult an agricultural expert
                    before making crop-treatment decisions.
                </div>
                """, unsafe_allow_html=True)

            else:
                st.info(
                    "Your image is ready. Select "
                    "'Analyse wheat image' to view the prediction."
                )

    except Exception as error:
        st.error(
            "Unable to process this image. Try a valid JPG, PNG, "
            "AVIF or WebP image."
        )
        st.caption(f"Technical details: {error}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    <b>WheatSense AI</b> · MobileNetV2 &amp; Deep Learning
    <br>
    An educational prototype for wheat image classification
</div>
""", unsafe_allow_html=True)