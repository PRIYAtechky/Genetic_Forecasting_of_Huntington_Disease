import streamlit as st
from PIL import Image
import re
import traceback

# Try importing predict safely
try:
    from predict_multimodal import predict
    PREDICT_AVAILABLE = True
    IMPORT_ERR = None
except Exception as e:
    PREDICT_AVAILABLE = False
    IMPORT_ERR = e

# ----- Page Config -----
st.set_page_config(
    page_title="Huntington's Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Sidebar -----
st.sidebar.title("🔬 HD Detection Controls")
CONF_THRESHOLD = st.sidebar.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
show_dna_input = st.sidebar.checkbox("Enable DNA Input in MRI Tab", True)
keep_history = st.sidebar.checkbox("Keep prediction history (this session)", True)

with st.sidebar.expander("ℹ️ Tips", expanded=False):
    st.markdown(
        "- MRI: upload a clear axial brain MRI image (PNG/JPG).\n"
        "- DNA: only letters **A/T/C/G**.\n"
        "- Low confidence → treat the result as **uncertain**.\n"
    )

# ----- Header -----
st.markdown(
    "<h1 style='text-align:center; color:#4B0082;'>🧬 Huntington's Disease Detection System</h1>",
    unsafe_allow_html=True
)

# ----- Session History -----
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(kind: str, cls: str, prob: float):
    if keep_history:
        st.session_state.history.append({"Type": kind, "Class": cls, "Confidence": f"{prob:.2%}"})

# ----- Helpers -----
DNA_REGEX = re.compile(r"^[ATCG]+$", re.IGNORECASE)
def is_valid_dna(seq): return bool(DNA_REGEX.fullmatch((seq or "").strip()))

def show_pred_block(title: str, cls: str, prob: float):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**{title}**")
        st.markdown(f"- Class: **{cls}**")
        st.markdown(f"- Confidence: **{prob:.2%}**")
    with c2:
        st.progress(min(1.0, float(prob)))
    if prob < CONF_THRESHOLD:
        st.warning("⚠ Low confidence. Result may be uncertain.")

def safe_predict(dna_text, image_file):
    if not PREDICT_AVAILABLE:
        st.error("❌ Could not import prediction module.")
        if IMPORT_ERR:
            with st.expander("See error details"):
                st.code(str(IMPORT_ERR))
        return None
    try:
        with st.spinner("Running prediction..."):
            return predict(dna_text, image_file)
    except Exception:
        st.error("❌ Error during prediction.")
        with st.expander("See error details"):
            st.code("".join(traceback.format_exc()))
        return None

# ----- Tabs -----
tabs = st.tabs(["🩻 MRI Prediction", "🧬 DNA Prediction"])

# MRI TAB
with tabs[0]:
    st.subheader("MRI Image Classification (3 classes)")
    uploaded = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

    img_ok = None
    if uploaded:
        try:
            img_ok = Image.open(uploaded).convert("RGB")
            st.image(img_ok, caption="Uploaded MRI", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Unable to open image. Error: {e}")
            img_ok = None

    dna_seq_optional = ""
    if img_ok is not None and show_dna_input:
        dna_seq_optional = st.text_area("Optional: Add DNA sequence (A/T/C/G only)")
        if dna_seq_optional and not is_valid_dna(dna_seq_optional):
            st.error("❌ Invalid DNA sequence! Only A/T/C/G characters are allowed.")
            dna_seq_optional = ""

    if img_ok is not None:
        results = safe_predict(dna_seq_optional if dna_seq_optional else "", uploaded)
        if results:
            mri_res = results.get("MRI")
            if isinstance(mri_res, dict) and "Class" in mri_res and "Probability" in mri_res:
                cls = str(mri_res["Class"]); prob = float(mri_res["Probability"])
                show_pred_block("MRI Prediction", cls, prob)
                add_history("MRI", cls, prob)
            else:
                st.error("❌ MRI result missing or malformed from predict().")

            if dna_seq_optional:
                dna_res = results.get("DNA")
                if isinstance(dna_res, dict) and "Class" in dna_res and "Probability" in dna_res:
                    st.markdown("---")
                    cls_d = str(dna_res["Class"]); prob_d = float(dna_res["Probability"])
                    show_pred_block("DNA Prediction (from MRI tab)", cls_d, prob_d)
                    add_history("DNA", cls_d, prob_d)
                else:
                    st.warning("DNA result not available or malformed. Check DNA input and model.")

# DNA TAB
with tabs[1]:
    st.subheader("DNA Sequence Classification (3 classes)")
    dna_input = st.text_area("Paste DNA sequence (A/T/C/G only)")
    if dna_input:
        if not is_valid_dna(dna_input):
            st.error("❌ Invalid DNA sequence! Only A/T/C/G characters are allowed.")
        else:
            results = safe_predict(dna_input, None)
            if results:
                dna_res = results.get("DNA")
                if isinstance(dna_res, dict) and "Class" in dna_res and "Probability" in dna_res:
                    cls = str(dna_res["Class"]); prob = float(dna_res["Probability"])
                    show_pred_block("DNA Prediction", cls, prob)
                    add_history("DNA", cls, prob)
                else:
                    st.error("❌ DNA result missing or malformed from predict().")

# History
if keep_history and st.session_state.history:
    st.markdown("---")
    st.subheader("🗂️ Session Prediction History")
    st.dataframe(st.session_state.history, use_container_width=True)

st.markdown(
    "<hr><p style='text-align:center; color:gray;'>© 2025 Huntington's Disease Detection</p>",
    unsafe_allow_html=True
)
