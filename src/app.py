import streamlit as st
from PIL import Image
import re
import traceback

# ==========================
# Import Prediction Function
# ==========================
try:
    from predict_multimodal import predict
    PREDICT_AVAILABLE = True
except Exception as e:
    PREDICT_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Huntington Disease DNA Detection",
    page_icon="🧬",
    layout="wide"
)

# ==========================
# Sidebar
# ==========================
st.sidebar.title("🧬 Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05
)

keep_history = st.sidebar.checkbox(
    "Keep Prediction History",
    value=True
)

# ==========================
# Session History
# ==========================
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# DNA Validation
# ==========================
DNA_REGEX = re.compile(r"^[ATCG]+$", re.IGNORECASE)

def is_valid_dna(seq):
    return bool(DNA_REGEX.fullmatch(seq.strip()))

# ==========================
# Header
# ==========================
st.markdown("""
<h1 style='text-align:center;color:#4B0082;'>
🧬 Huntington's Disease Detection Using DNA Sequence
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================
# Input Section
# ==========================
st.subheader("Upload MRI and Enter DNA Sequence")

uploaded_file = st.file_uploader(
    "Upload MRI Image (Optional)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, width=250)
        st.success("✅ MRI uploaded successfully (Used only for display)")
    except:
        st.error("❌ Invalid image file")

dna_sequence = st.text_area(
    "Paste DNA Sequence (A/T/C/G only)",
    height=150
)

# ==========================
# Prediction Function
# ==========================
def run_prediction(dna_text):

    if not PREDICT_AVAILABLE:
        st.error("❌ predict_multimodal.py could not be imported")
        st.code(IMPORT_ERROR)
        return

    try:
        with st.spinner("Analyzing DNA Sequence..."):

            result = predict(dna_text, None)

            if "DNA" not in result:
                st.error("❌ DNA result not found")
                return

            dna_result = result["DNA"]

            pred_class = dna_result["Class"]
            confidence = float(dna_result["Probability"])

            st.markdown("---")
            st.subheader("📋 Final Prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Class", pred_class)
                st.metric(
                    "Confidence Score",
                    f"{confidence*100:.2f}%"
                )

            with col2:
                st.progress(confidence)

            if confidence < confidence_threshold:
                st.warning(
                    "⚠ Low confidence prediction. Verify with medical experts."
                )

            if pred_class.lower() == "normal":
                st.success("✅ DNA appears Normal")

            elif pred_class.lower() == "intermediate":
                st.warning("⚠ Intermediate Risk Detected")

            elif pred_class.lower() == "pathogenic":
                st.error("🚨 Pathogenic Pattern Detected")

            if keep_history:
                st.session_state.history.append({
                    "Prediction": pred_class,
                    "Confidence": f"{confidence*100:.2f}%"
                })

    except Exception:
        st.error("Prediction Error")
        st.code(traceback.format_exc())

# ==========================
# Predict Button
# ==========================
if st.button("🔍 Find Output", use_container_width=True):

    if dna_sequence.strip() == "":
        st.warning("⚠ Please enter DNA sequence")
        st.stop()

    if not is_valid_dna(dna_sequence):
        st.error(
            "❌ Invalid DNA sequence.\nOnly A, T, C and G are allowed."
        )
        st.stop()

    run_prediction(dna_sequence)

# ==========================
# History Section
# ==========================
if keep_history and len(st.session_state.history) > 0:

    st.markdown("---")
    st.subheader("🗂 Prediction History")

    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown(
    "<center>© 2026 Huntington Disease DNA Detection System</center>",
    unsafe_allow_html=True
)
