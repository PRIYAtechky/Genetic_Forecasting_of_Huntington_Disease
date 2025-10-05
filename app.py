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
keep_history = st.sidebar.checkbox("Keep prediction history (this session)", True)

with st.sidebar.expander("ℹ️ Tips", expanded=False):
    st.markdown(
        "- Upload an axial brain MRI image (PNG/JPG).\n"
        "- Paste a valid DNA sequence (only A/T/C/G).\n"
        "- Click **Find Output** to get the combined prediction.\n"
    )

# ----- Header -----
st.markdown(
    "<h1 style='text-align:center; color:#4B0082;'>🧬 Huntington's Disease Combined Detection</h1>",
    unsafe_allow_html=True
)

# ----- Session History -----
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(cls: str, prob: float):
    if keep_history:
        st.session_state.history.append({"Class": cls, "Confidence": f"{prob:.2%}"})


# ----- Helpers -----
DNA_REGEX = re.compile(r"^[ATCG]+$", re.IGNORECASE)
def is_valid_dna(seq): return bool(DNA_REGEX.fullmatch((seq or "").strip()))

def show_pred_block(cls: str, prob: float):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 🧾 Final Combined Prediction")
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

# ----- Combine Function -----
def combine_results(mri_res, dna_res):
    if not mri_res or not dna_res:
        return None
    
    cls_m, prob_m = mri_res["Class"], float(mri_res["Probability"])
    cls_d, prob_d = dna_res["Class"], float(dna_res["Probability"])

    if cls_m == cls_d:
        final_cls = cls_m
        final_prob = (prob_m + prob_d) / 2
    else:
        if prob_m >= prob_d:
            final_cls, final_prob = cls_m, prob_m
        else:
            final_cls, final_prob = cls_d, prob_d
    
    return {"Class": final_cls, "Probability": final_prob}

# ----- Input Section -----
st.subheader("Upload MRI and Enter DNA Sequence")

uploaded = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])
dna_input = st.text_area("Paste DNA sequence (A/T/C/G only)")

img_ok = None
if uploaded:
    try:
        img_ok = Image.open(uploaded).convert("RGB")
        st.image(img_ok, caption="Uploaded MRI", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Unable to open image. Error: {e}")
        img_ok = None

# ----- Find Output Button -----
if st.button("🔍 Find Output"):
    if uploaded and dna_input:
        if not is_valid_dna(dna_input):
            st.error("❌ Invalid DNA sequence! Only A/T/C/G characters are allowed.")
        else:
            results = safe_predict(dna_input, uploaded)
            if results:
                mri_res = results.get("MRI")
                dna_res = results.get("DNA")

                if mri_res and dna_res:
                    combined = combine_results(mri_res, dna_res)
                    if combined:
                        cls_c, prob_c = combined["Class"], combined["Probability"]
                        show_pred_block(cls_c, prob_c)
                        add_history(cls_c, prob_c)
                else:
                    st.error("❌ MRI or DNA result missing from predict().")
    else:
        st.warning("⚠ Please upload an MRI and paste a DNA sequence before clicking Find Output.")

# ----- History -----
if keep_history and st.session_state.history:
    st.markdown("---")
    st.subheader("🗂️ Session Prediction History")
    st.dataframe(st.session_state.history, use_container_width=True)

st.markdown(
    "<hr><p style='text-align:center; color:gray;'>© 2025 Huntington's Disease Detection</p>",
    unsafe_allow_html=True
)
