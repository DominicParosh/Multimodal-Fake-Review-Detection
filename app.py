"""
Streamlit App — Multimodal Fake Review Detector
================================================

Usage:
    streamlit run app.py
"""

import sys
import os
from typing import Optional

import streamlit as st
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# Force Light Theme via CSS
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Force light background everywhere ── */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"], .main, section[data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #f7f8fa !important;
        color: #1a1a2e !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* ── Force all text dark ── */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, td, th,
    .stMarkdown, .stText, [data-testid="stSidebar"] * {
        color: #1a1a2e !important;
    }

    /* ── Text area and inputs ── */
    textarea, input, [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 2px solid #d1d5db !important;
        border-radius: 10px !important;
    }
    textarea:focus, input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    }

    /* ── Select boxes ── */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 2px solid #d1d5db !important;
        border-radius: 10px !important;
    }
    [data-baseweb="popover"] li {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }
    [data-baseweb="popover"] li:hover {
        background-color: #f0f0ff !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 2px dashed #c7c7d1 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] * {
        color: #1a1a2e !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"], .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
    }

    /* ── Alert / Info box ── */
    [data-testid="stAlert"] {
        background-color: #eff6ff !important;
        border: 1px solid #93c5fd !important;
        border-radius: 12px !important;
    }
    [data-testid="stAlert"] * {
        color: #1e40af !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] * { color: #1a1a2e !important; }

    /* ── Metrics ── */
    [data-testid="stMetric"] { background: #ffffff !important; border-radius: 12px !important; padding: 0.5rem !important; }

    /* ── CUSTOM COMPONENTS ── */

    .app-header {
        text-align: center;
        padding: 1.2rem 0 0.8rem;
        margin-bottom: 1.2rem;
        border-bottom: 3px solid #6366f1;
    }
    .app-header h1 { font-weight: 800 !important; font-size: 2.4rem !important; color: #1a1a2e !important; margin: 0 !important; }
    .app-header .sub { color: #6b7280 !important; font-size: 1rem !important; margin-top: 4px !important; }

    .result-card {
        border-radius: 16px;
        padding: 1.8rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .result-card-fake {
        border: 2px solid #ef4444; border-left: 6px solid #ef4444;
        background: #fef2f2 !important;
    }
    .result-card-real {
        border: 2px solid #22c55e; border-left: 6px solid #22c55e;
        background: #f0fdf4 !important;
    }

    .verdict { font-weight: 800 !important; font-size: 2rem !important; margin-bottom: 0.3rem; }
    .verdict-fake { color: #dc2626 !important; }
    .verdict-real { color: #16a34a !important; }

    .conf-bar-bg {
        width: 100%; height: 14px;
        background: #e5e7eb; border-radius: 7px;
        overflow: hidden; margin-top: 0.6rem;
    }
    .conf-bar-fake { height:100%; border-radius:7px; background: linear-gradient(90deg, #fca5a5, #ef4444); }
    .conf-bar-real { height:100%; border-radius:7px; background: linear-gradient(90deg, #bbf7d0, #22c55e); }
    .conf-label { font-size: 1.4rem !important; font-weight: 700 !important; margin-top: 0.4rem; }

    .metric-box {
        text-align: center; padding: 1.2rem;
        border-radius: 14px; border: 2px solid #e5e7eb;
        background: #ffffff !important;
    }
    .metric-box .value { font-size: 1.8rem !important; font-weight: 800 !important; font-family: monospace; }
    .metric-box .label { font-size: 0.78rem !important; color: #6b7280 !important; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 4px; }

    .explanation-box {
        background: #ffffff !important; border: 2px solid #e2e8f0;
        border-radius: 14px; padding: 1.3rem 1.6rem;
        margin-top: 0.8rem; font-size: 0.95rem; line-height: 1.7;
        color: #334155 !important;
    }

    .token-chip {
        display: inline-block; padding: 5px 14px; margin: 4px;
        border-radius: 20px; font-family: monospace;
        font-size: 0.82rem; font-weight: 700;
    }
    .token-high  { background: #fee2e2 !important; color: #991b1b !important; border: 1.5px solid #fca5a5; }
    .token-med   { background: #fef3c7 !important; color: #92400e !important; border: 1.5px solid #fcd34d; }
    .token-low   { background: #e0f2fe !important; color: #075985 !important; border: 1.5px solid #7dd3fc; }

    .meter-track {
        flex: 1; height: 10px;
        background: linear-gradient(90deg, #ef4444, #eab308, #22c55e);
        border-radius: 5px; position: relative;
    }
    .meter-thumb {
        width: 18px; height: 18px;
        background: #1a1a2e; border: 3px solid #ffffff;
        border-radius: 50%; position: absolute; top: -4px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }

    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #6366f1);
        border: none; border-radius: 2px; margin: 2rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────

@st.cache_resource
def load_detector(checkpoint_path: str, config_path: str, device: str = "cpu"):
    try:
        from toolkit.detector import FakeReviewDetector
        detector = FakeReviewDetector.from_pretrained(checkpoint_path, config_path, device)
        return detector, True
    except Exception:
        return None, False


def run_mock_prediction(text: str, image: Optional[Image.Image] = None) -> dict:
    """Heuristic-based mock prediction for demo mode."""
    import hashlib

    signals = []
    text_lower = text.lower()

    exclaim_ratio = text.count("!") / max(len(text), 1)
    signals.append(min(exclaim_ratio * 20, 1.0))

    superlatives = ["best", "amazing", "incredible", "perfect", "worst", "terrible",
                    "fantastic", "awful", "love", "hate", "horrible"]
    sup_count = sum(1 for w in superlatives if w in text_lower)
    signals.append(min(sup_count * 0.25, 1.0))

    word_count = len(text.split())
    signals.append(0.6 if word_count < 8 or word_count > 200 else 0.2)

    words = text_lower.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    signals.append(max(0, 1.0 - unique_ratio))

    generic = ["highly recommend", "must buy", "life changing", "game changer",
               "do not buy", "waste of money", "five stars", "absolutely love",
               "total garbage", "best purchase ever"]
    gen_count = sum(1 for g in generic if g in text_lower)
    signals.append(min(gen_count * 0.3, 1.0))

    image_factor = -0.1 if image is not None else 0.05
    raw_score = np.mean(signals) + image_factor
    fake_prob = max(0.05, min(0.95, raw_score + 0.15))

    text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    noise = (text_hash % 100 - 50) / 500.0
    fake_prob = max(0.05, min(0.95, fake_prob + noise))

    is_fake = fake_prob > 0.5
    confidence = fake_prob if is_fake else (1 - fake_prob)

    suspicious_words = []
    for word in words:
        if word in superlatives:
            score = 0.85
        elif word in ["recommend", "buy", "perfect", "horrible", "amazing"]:
            score = 0.7
        elif word.endswith("!"):
            score = 0.6
        elif word in ["the", "a", "is", "it", "and", "to", "i", "my", "for"]:
            score = 0.05
        else:
            score = (hash(word) % 40) / 100.0
        suspicious_words.append({"token": word, "score": round(score, 4)})
    suspicious_words.sort(key=lambda x: x["score"], reverse=True)

    consistency = round(0.3 + np.random.RandomState(text_hash).random() * 0.5, 4)
    if image is None:
        consistency = 0.5

    top_words = suspicious_words[:8]
    expl = f"The review was classified as {'FAKE' if is_fake else 'GENUINE'} with {confidence:.1%} confidence."
    if is_fake:
        flagged = [w["token"] for w in top_words[:3] if w["score"] > 0.5]
        if flagged:
            expl += f" Suspicious language detected: {', '.join(flagged)}."
        if image is not None and consistency < 0.4:
            expl += " The uploaded image appears inconsistent with the review text."
    else:
        expl += " The language patterns appear consistent with authentic customer reviews."
    if image is None:
        expl += " No image was provided; text-only analysis was performed."

    return {
        "prediction": 1 if is_fake else 0,
        "label": "fake" if is_fake else "real",
        "confidence": round(confidence, 4),
        "probabilities": {"real": round(1 - fake_prob, 4), "fake": round(fake_prob, 4)},
        "explanation": {
            "text": expl,
            "top_influential_words": top_words,
            "text_image_consistency": consistency,
        },
    }


# ─────────────────────────────────────────────────────
# Examples
# ─────────────────────────────────────────────────────

EXAMPLES = {
    "Select an example...": "",
    "✅ Genuine — Balanced Review": (
        "I've been using these headphones for about 3 weeks now. The sound quality is good for "
        "the price point, though the bass could be a bit stronger. Comfort is decent for short "
        "sessions but they get a little warm after an hour. Battery life is as advertised. "
        "Overall solid purchase for $40."
    ),
    "🚩 Fake — Overly Enthusiastic": (
        "OMG this is the BEST product I have EVER purchased in my entire life!!! "
        "Absolutely incredible, life-changing, must-buy!! I recommend this to everyone!! "
        "Five stars is not enough!! Perfect perfect perfect!!"
    ),
    "🚩 Fake — Generic Shill": (
        "This product is amazing and I highly recommend it to anyone looking for a great deal. "
        "The quality is fantastic and the price is unbeatable. Must buy for sure. "
        "You will not be disappointed. Best purchase ever."
    ),
    "✅ Genuine — Detailed Critique": (
        "Bought the air purifier for my 200 sq ft bedroom. CADR rating seemed good on paper. "
        "In practice, it reduced dust noticeably within a week but didn't help much with cooking "
        "smells from the kitchen. The night mode is quiet enough to sleep through. Filter "
        "replacement is $30 every 6 months which adds up. Fine for allergies, not great for odors."
    ),
    "⚠️ Suspicious — Short & Vague": "Great product. Works well. Would buy again.",
}


# ─────────────────────────────────────────────────────
# UI Rendering
# ─────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="app-header">
        <h1>🔍 Fake Review Detector</h1>
        <p class="sub">Multimodal AI analysis — paste review text and/or upload an image</p>
    </div>
    """, unsafe_allow_html=True)


def render_verdict(result):
    is_fake = result["label"] == "fake"
    card = "result-card-fake" if is_fake else "result-card-real"
    vcls = "verdict-fake" if is_fake else "verdict-real"
    vtxt = "⚠️  LIKELY FAKE" if is_fake else "✅  LIKELY GENUINE"
    bar = "conf-bar-fake" if is_fake else "conf-bar-real"
    pct = result["confidence"] * 100
    clr = "#dc2626" if is_fake else "#16a34a"
    st.markdown(f"""
    <div class="result-card {card}">
        <div class="verdict {vcls}">{vtxt}</div>
        <div class="conf-bar-bg"><div class="{bar}" style="width:{pct}%"></div></div>
        <div class="conf-label" style="color:{clr} !important">{pct:.1f}% confidence</div>
    </div>""", unsafe_allow_html=True)


def render_probabilities(result):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="metric-box">
            <div class="value" style="color:#16a34a !important">{result['probabilities']['real']:.1%}</div>
            <div class="label">Real Probability</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <div class="value" style="color:#dc2626 !important">{result['probabilities']['fake']:.1%}</div>
            <div class="label">Fake Probability</div></div>""", unsafe_allow_html=True)


def render_explanation(explanation, has_image):
    st.markdown("#### 💡 Explanation")
    st.markdown(f'<div class="explanation-box">{explanation["text"]}</div>', unsafe_allow_html=True)

    if explanation.get("top_influential_words"):
        st.markdown("")
        st.markdown("**Key Words Influencing Detection:**")
        html = ""
        for t in explanation["top_influential_words"][:10]:
            s = t["score"]
            cls = "token-high" if s > 0.6 else ("token-med" if s > 0.3 else "token-low")
            html += f'<span class="token-chip {cls}">{t["token"]} ({s:.2f})</span>'
        st.markdown(f'<div style="margin:0.5rem 0 1rem">{html}</div>', unsafe_allow_html=True)

    if has_image and "text_image_consistency" in explanation:
        cons = explanation["text_image_consistency"]
        pos = cons * 100
        if cons < 0.35:
            lbl, lc = "Low — text and image seem mismatched", "#dc2626"
        elif cons < 0.65:
            lbl, lc = "Moderate", "#ca8a04"
        else:
            lbl, lc = "High — text matches the image well", "#16a34a"
        st.markdown("**Text-Image Consistency:**")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:0.6rem 0">
            <span style="font-size:0.8rem;color:#ef4444 !important;font-weight:600">Low</span>
            <div class="meter-track"><div class="meter-thumb" style="left:calc({pos}% - 9px)"></div></div>
            <span style="font-size:0.8rem;color:#22c55e !important;font-weight:600">High</span>
        </div>
        <div style="text-align:center;font-size:0.92rem;color:{lc} !important;font-weight:600">{cons:.2f} — {lbl}</div>
        """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        show_explanation = st.checkbox("Show Explanation", value=True)
        show_tokens = st.checkbox("Show Token Analysis", value=True)
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown(
            "This tool uses a **multimodal AI model** combining:\n"
            "- 🔤 **RoBERTa** text encoder\n"
            "- 🖼️ **CLIP ViT** image encoder\n"
            "- 🔀 **Cross-modal attention** fusion\n\n"
            "It analyzes review text and images together to detect fake or AI-generated reviews."
        )
        st.markdown("---")
        st.markdown("### 🎯 Detection Cues")
        st.markdown(
            "The model looks for:\n"
            "- Unnatural language patterns\n"
            "- Text-image mismatches\n"
            "- Suspicious metadata signals\n"
            "- AI generation artifacts"
        )
        return {"device": device, "threshold": threshold,
                "show_explanation": show_explanation, "show_tokens": show_tokens}


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    render_header()
    settings = render_sidebar()

    model_loaded = False
    detector = None
    ckpt = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_model.pt")
    cfg = os.environ.get("CONFIG_PATH", "configs/config.yaml")

    if os.path.exists(ckpt) and os.path.exists(cfg):
        detector, model_loaded = load_detector(ckpt, cfg, settings["device"])

    if not model_loaded:
        st.info(
            "🧪 **Demo Mode** — No trained model found. Using heuristic-based simulation. "
            "Train a model first, then place `best_model.pt` in `checkpoints/` for real inference."
        )

    # ── Input ──
    st.markdown("### 📝 Input Review")
    example_choice = st.selectbox("Try an example:", list(EXAMPLES.keys()))

    col_text, col_image = st.columns([3, 2])
    with col_text:
        review_text = st.text_area("Review Text", value=EXAMPLES[example_choice],
                                   height=200, placeholder="Paste the review text here...")
    with col_image:
        uploaded = st.file_uploader("Review Image (optional)", type=["jpg","jpeg","png","webp"],
                                    help="Upload the image attached to the review")
        pil_image = None
        if uploaded:
            pil_image = Image.open(uploaded).convert("RGB")
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

    st.markdown("")
    if st.button("🔍  Analyze Review", type="primary", use_container_width=True):
        if not review_text.strip() and pil_image is None:
            st.warning("Please provide review text and/or an image.")
            return

        with st.spinner("Analyzing..."):
            if model_loaded and detector:
                result = detector.predict(text=review_text.strip() or "No text",
                                          image=pil_image, explain=settings["show_explanation"])
                if "explanation" not in result:
                    result["explanation"] = {"text": f"Classified as {result['label'].upper()} with {result['confidence']:.1%} confidence.",
                                             "top_influential_words": [], "text_image_consistency": 0.5}
            else:
                result = run_mock_prediction(review_text.strip(), pil_image)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Detection Results")
        render_verdict(result)
        st.markdown("")
        render_probabilities(result)
        if settings["show_explanation"] and "explanation" in result:
            st.markdown("")
            render_explanation(result["explanation"], has_image=pil_image is not None)
        with st.expander("🔧 Raw Model Output"):
            st.json(result)

    # ── Batch ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📋 Batch Analysis")
    batch_text = st.text_area("Paste multiple reviews (one per line)", height=120,
                              placeholder="Review 1\nReview 2\nReview 3", key="batch_input")
    if st.button("🔍  Analyze Batch", key="batch_btn"):
        if batch_text.strip():
            reviews = [r.strip() for r in batch_text.strip().split("\n") if r.strip()][:50]
            progress = st.progress(0)
            results_list = []
            for i, rev in enumerate(reviews):
                res = (detector.predict(text=rev, explain=False) if model_loaded and detector
                       else run_mock_prediction(rev))
                res["text_preview"] = rev[:80] + ("..." if len(rev) > 80 else "")
                results_list.append(res)
                progress.progress((i + 1) / len(reviews))
            progress.empty()

            fake_count = sum(1 for r in results_list if r["label"] == "fake")
            avg_conf = np.mean([r["confidence"] for r in results_list])
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Reviews", len(results_list))
            c2.metric("Flagged Fake", fake_count)
            c3.metric("Avg Confidence", f"{avg_conf:.1%}")

            table = [{"Status": f"{'🔴' if r['label']=='fake' else '🟢'} {r['label'].upper()}",
                       "Confidence": f"{r['confidence']:.1%}",
                       "P(Fake)": f"{r['probabilities']['fake']:.3f}",
                       "Review": r["text_preview"]} for r in results_list]
            st.dataframe(table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()