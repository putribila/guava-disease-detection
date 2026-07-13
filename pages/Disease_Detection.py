from utils.gradcam import overlay_heatmap
from utils.gradcam import generate_gradcam
from utils.predictor import get_model
from utils.disease_info import get_disease_info
import time
import pandas as pd
import streamlit as st
from PIL import Image

from utils.preprocessing import preprocess_image
from utils.predictor import (
    predict,
    class_names,
    is_guava
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Disease Detection",
    page_icon="🍈",
    layout="wide"
)

# ==========================
# LOAD CSS
# ==========================

with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ==========================
# HEADER
# ==========================

st.markdown("""
<div class="main-title">
🍈 Guava Disease Detector AI
</div>

<div class="subtitle">
Deep Learning Based Guava Disease Classification using MobileNetV2
</div>
""",
unsafe_allow_html=True)
st.markdown("---")

st.write(
"""
Upload an image of a **guava fruit** to classify it into one of the following classes:

- Healthy
- Anthracnose
- Fruit Fly
"""
)

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# START PREDICTION
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with st.spinner("Validating image..."):

        if not is_guava(image):

            st.error(
                "❌ The uploaded image is not recognized as a guava fruit. Please upload another image."
            )

            st.stop()

    processed_image = preprocess_image(image)

    start = time.time()

    predicted_class, confidence, prediction = predict(processed_image)

    disease_info = get_disease_info(predicted_class)

    inference_time = time.time() - start

    # ==========================
    # Generate Grad-CAM
    # ==========================

    model = get_model()

    heatmap = generate_gradcam(
        model,
        processed_image
    )

    gradcam_image = overlay_heatmap(
        image,
        heatmap
    )
    # =====================================================

    col1, col2 = st.columns([1,1])

    with col1:

        st.markdown("## 📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown(f"""
        <div class="prediction-card">
            <h3>Prediction Result</h3>
            <h1>{predicted_class}</h1>
            <h2>{confidence*100:.2f}%</h2>
        </div>
        """,
        unsafe_allow_html=True)

        st.markdown("")

        st.metric(

            label="Confidence",

            value=f"{confidence*100:.2f}%"

        )

        st.metric(

            label="Inference Time",

            value=f"{inference_time:.4f} sec"

        )

    st.markdown("---")

    st.markdown("# 📖 Disease Information")

    st.markdown(
        f"""
        <div class="card">
            <h3 style="margin: 0 0 4px 0; color: #1B4332;">{disease_info['title']}</h3>
            <p style="margin: 0 0 10px 0; font-size: 15px;"><b>Severity:</b> {disease_info['severity']}</p>
            <p style="margin: 0 0 4px 0; font-size: 15px;"><b>Description:</b></p>
            <p style="margin: 0; font-size: 14px; color: #495057; line-height: 1.5;">{disease_info['description']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Symptoms")

        for symptom in disease_info["symptoms"]:

            st.success(symptom)

    with col2:

        st.markdown("### Recommendation")

        for rec in disease_info["recommendation"]:

            st.info(rec)

    st.markdown("### Cause")

    st.warning(disease_info["cause"])
    # ======================================================
    # EXPLAINABLE AI
    # ======================================================

    st.markdown("---")

    st.markdown("## 🔥 Explainable AI (Grad-CAM)")

    st.write(
        """
        The heatmap below highlights the image regions that most influenced the model's prediction.
        Red and yellow areas indicate regions receiving higher attention from the model.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### Grad-CAM")

        st.image(
            gradcam_image,
            use_container_width=True
        )

    st.info(
        """
        🧠 **Interpretation**

        Grad-CAM helps visualize which regions of the guava fruit the MobileNetV2 model focused on during classification.

        - 🔴 Red → Highest attention
        - 🟠 Orange → Moderate attention
        - 🔵 Blue → Low attention
        """,
    )

    # =======================================================
    # PROBABILITY
    # =======================================================

    st.markdown("## 📊 Probability Distribution")

    df = pd.DataFrame({

        "Disease": class_names,

        "Probability": prediction

    })

    df["Probability"] *= 100

    df = df.sort_values(

        by="Probability",

        ascending=False

    )

    for _, row in df.iterrows():

        st.write(f"**{row['Disease']}**")

        st.progress(

            float(row["Probability"]/100)

        )

        st.caption(

            f"{row['Probability']:.2f}%"

        )

    st.markdown("---")

    # =======================================================
    # IMAGE INFO
    # =======================================================

    st.markdown("## 🖼 Image Information")

    info1, info2 = st.columns(2)

    with info1:

        st.metric(

            "Resolution",

            f"{image.size[0]} × {image.size[1]}"

        )

    with info2:

        st.metric(

            "Image Mode",

            image.mode

        )

    st.markdown("---")

    # =======================================================
    # MODEL INFO
    # =======================================================

    st.markdown("## 🤖 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Architecture",

            "MobileNetV2"

        )

    with col2:

        st.metric(

            "Input Size",

            "224 × 224"

        )

    with col3:

        st.metric(

            "Classes",

            len(class_names)

        )

    st.markdown("""

<div class="footer">

Guava Disease Detector AI • Built with TensorFlow • Streamlit

</div>

""",
unsafe_allow_html=True)