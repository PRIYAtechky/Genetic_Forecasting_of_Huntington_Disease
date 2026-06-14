import streamlit as st
from PIL import Image
import re
import traceback

from predict_multimodal import predict

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Huntington Disease Detection",
    page_icon="🧬",
    layout="wide"
)

# =====================================
# DARK PROFESSIONAL THEME
# =====================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0f172a;
    color: white;
}

/* Container */
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Title */
h1 {
    text-align: center;
    color: #ffffff !important;
    font-size: 48px !important;
    font-weight: 800;
}

/* Headers */
h2, h3 {
    color: #60a5fa !important;
    font-size: 30px !important;
}

/* All Text */
p, li, span, label {
    color: white !important;
    font-size: 18px !important;
}

/* Upload Card */
[data-testid="stFileUploader"] {
    background: #1e293b;
    border-radius: 15px;
    padding: 15px;
    border: 1px solid #334155;
}

/* Text Area */
.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #475569 !important;
    font-size: 18px !important;
}

/* Predict Button */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        90deg,
        #2563eb,
        #3b82f6
    );
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #2563eb
    );
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

/* Metric Text */
[data-testid="stMetricLabel"] {
    color: white !important;
}

[data-testid="stMetricValue"] {
    color: #60a5fa !important;
}

/* Expander */
details {
    background: #1e293b;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #334155;
}

/* Success */
.stSuccess {
    border-radius: 10px;
}

/* Warning */
.stWarning {
    border-radius: 10px;
}

/* Error */
.stError {
    border-radius: 10px;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.title("🧬 Huntington Disease Detection System")

st.markdown("""
### Multimodal  Classification

This system combines:

- 🧬 DNA Analysis
- 🧠 MRI Analysis

to predict:

- Normal
- Intermediate
- Pathogenic
""")

# =====================================
# VALIDATION
# =====================================

DNA_REGEX = re.compile(
    r"^[ATCG]+$",
    re.IGNORECASE
)

# =====================================
# MRI UPLOAD
# =====================================

st.subheader("🧠 MRI Upload")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg"]
)

image = None

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        st.image(
            image,
            width=350,
            caption="Uploaded MRI Image"
        )

        st.success(
            "✅ MRI uploaded successfully"
        )

    except Exception:

        st.error(
            "❌ Unable to open image"
        )

# =====================================
# DNA INPUT
# =====================================

st.subheader("🧬 DNA Sequence")

dna_sequence = st.text_area(
    "Enter DNA Sequence",
    height=200,
    placeholder="ATCGATCGATCGATCG..."
)

# =====================================
# PREDICTION
# =====================================

if st.button("🔍 Predict"):

    if uploaded_file is None:

        st.warning(
            "⚠ Please upload MRI image."
        )

        st.stop()

    if not dna_sequence.strip():

        st.warning(
            "⚠ Please enter DNA sequence."
        )

        st.stop()

    if not DNA_REGEX.fullmatch(
        dna_sequence.strip()
    ):

        st.error(
            "❌ Only A, T, C and G are allowed."
        )

        st.stop()

    try:

        with st.spinner(
            "Analyzing DNA and MRI..."
        ):

            result = predict(
                dna_sequence,
                image
            )

        dna_result = result["DNA"]
        mri_result = result["MRI"]
        final_result = result["FINAL"]

        st.markdown("---")

        st.subheader(
            "📊 Multimodal Prediction Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🧬 DNA Result",
                dna_result["Class"],
                f"{dna_result['Probability']:.2%}"
            )

        with col2:

            st.metric(
                "🧠 MRI Result",
                mri_result["Class"],
                f"{mri_result['Probability']:.2%}"
            )

        with col3:

            st.metric(
                "🎯 Final Result",
                final_result["Class"],
                f"{final_result['Probability']:.2%}"
            )

        st.progress(
            float(
                final_result["Probability"]
            )
        )

        # =====================================
        # FINAL DIAGNOSIS
        # =====================================

        if final_result["Class"] == "Normal":

            st.success(
                "✅ Final Diagnosis: NORMAL"
            )

        elif final_result["Class"] == "Intermediate":

            st.warning(
                "⚠ Final Diagnosis: INTERMEDIATE"
            )

        else:

            st.error(
                "🚨 Final Diagnosis: PATHOGENIC"
            )

        # =====================================
        # DETAILS
        # =====================================

        with st.expander(
            "🔍 Prediction Details"
        ):

            st.subheader(
                "DNA Prediction"
            )

            st.json(
                dna_result
            )

            st.subheader(
                "MRI Prediction"
            )

            st.json(
                mri_result
            )

            st.subheader(
                "Final Prediction"
            )

            st.json(
                final_result
            )

    except Exception:

        st.error(
            "❌ Prediction Error"
        )

        st.code(
            traceback.format_exc()
        )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "© 2026 Huntington Disease Detection System"
)
