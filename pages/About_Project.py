import streamlit as st

# =====================
# CSS
# =====================

with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =====================
# HEADER
# =====================

st.markdown("""
<div class="main-title">
🍈 About Project
</div>

<div class="subtitle">
Explainable AI Web Application for Guava Fruit Disease Classification
</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

st.markdown("## 📌 Project Overview")

st.write(
"""
Guava Disease Detector AI is a web-based application developed to
automatically classify guava fruit diseases using Deep Learning.

The application utilizes MobileNetV2 as the backbone CNN architecture
to classify guava fruits into three categories:

- Healthy
- Anthracnose
- Fruit Fly

In addition to disease prediction, the application also provides
Explainable AI visualization using Grad-CAM to help users understand
which image regions influenced the model's decision.
"""
)

st.divider()

# ==========================================================
# PROBLEM
# ==========================================================

st.markdown("## 🎯 Problem Statement")

st.info("""
Early identification of guava diseases is essential to reduce crop losses.
Manual inspection requires experience and is often time-consuming.

This project aims to assist users by providing an automatic image-based
classification system using Deep Learning.
""")

st.divider()

# ==========================================================
# WORKFLOW
# ==========================================================

st.markdown("## ⚙️ Project Workflow")

st.markdown("""
```text
Image Upload
      │
      ▼
Image Preprocessing
      │
      ▼
MobileNetV2
      │
      ▼
Disease Prediction
      │
      ▼
Grad-CAM
      │
      ▼
Disease Information
""")
# ==========================================================
# DATASET
# ==========================================================

st.divider()

st.markdown("## 📂 Dataset")

st.write("""
The dataset consists of guava fruit images collected from various conditions and manually categorized into three disease classes. Before training, all images underwent preprocessing, including resizing, normalization, and data augmentation to improve the model's generalization performance.
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📷 Total Images",
        value="3,784"
    )

with col2:
    st.metric(
        label="🦠 Classes",
        value="3"
    )

with col3:
    st.metric(
        label="🖼 Input Size",
        value="224 × 224"
    )

with col4:
    st.metric(
        label="🎨 Color Mode",
        value="RGB"
    )

st.markdown("### Dataset Classes")

dataset_df = {
    "Class": [
        "Anthracnose",
        "Fruit Fly",
        "Healthy"
    ],
    "Description": [
        "Fungal disease causing black sunken lesions on the fruit.",
        "Damage caused by fruit fly infestation.",
        "Healthy guava fruit without disease symptoms."
    ]
}

import pandas as pd

st.dataframe(
    pd.DataFrame(dataset_df),
    use_container_width=True,
    hide_index=True
)

st.info(
    """
**Dataset Summary**

• Total Images : **3,784**

• Image Resolution : **224 × 224 pixels**

• Number of Classes : **3**

• Task : **Image Classification**

• Data Split : **Train / Validation / Test**
"""
)

# ==========================================================
# MODEL
# ==========================================================

st.divider()

st.markdown("## 🧠 Model Architecture")

st.write("""
The disease classification model is built using **MobileNetV2** with the
Transfer Learning approach. MobileNetV2 was selected because it provides
high classification performance while remaining lightweight and suitable
for real-time deployment in web applications.
""")

st.markdown("### 📈 Model Summary")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Architecture",
        "MobileNetV2"
    )

with m2:
    st.metric(
        "Input Size",
        "224×224"
    )

with m3:
    st.metric(
        "Classes",
        "3"
    )

with m4:
    st.metric(
        "Accuracy",
        "95.81%"
    )

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### 🔹 Model Information

- **Architecture** : MobileNetV2
- **Framework** : TensorFlow & Keras
- **Learning Method** : Transfer Learning
- **Input Size** : 224 × 224 × 3
- **Output Classes** : 3
- **Activation** : Softmax
- **Optimizer** : Adam
- **Loss Function** : Categorical Crossentropy
    """)

with col2:

    st.markdown("""
### 📊 Model Performance

- **Test Accuracy** : **95.81%**
- **Precision** : **95.79%**
- **Recall** : **95.80%**
- **F1-Score** : **95.77%**
- **Explainable AI** : Grad-CAM
    """)

st.success("""
The final model was selected from four experimental scenarios and achieved
the highest classification performance while maintaining good generalization
on unseen test data.
""")

# ==========================================================
# TECHNOLOGY STACK
# ==========================================================

st.divider()

st.markdown("## 💻 Technology Stack")

st.write("""
This project was developed using modern technologies for deep learning,
image processing, web application development, and data visualization.
Each technology plays an important role in building an end-to-end AI system.
""")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("### 🤖 Artificial Intelligence")

    st.success("TensorFlow")
    st.success("Keras")
    st.success("MobileNetV2")
    st.success("NumPy")

with col2:

    st.markdown("### 🖼 Computer Vision")

    st.success("OpenCV")
    st.success("Pillow (PIL)")
    st.success("Grad-CAM")
    st.success("Image Processing")

with col3:

    st.markdown("### 🌐 Web Development")

    st.success("Streamlit")
    st.success("Pandas")
    st.success("Plotly")
    st.success("Python")

st.divider()

st.markdown("### 🏗 Development Workflow")

workflow_col1, workflow_col2 = st.columns(2)

with workflow_col1:

    st.markdown("""
**Model Development**

- Dataset Preparation
- Image Preprocessing
- Data Augmentation
- Transfer Learning
- Model Training
- Model Evaluation
""")

with workflow_col2:

    st.markdown("""
**Application Development**

- Streamlit Dashboard
- Disease Detection
- Explainable AI (Grad-CAM)
- Model Performance Dashboard
- Disease Information
- Interactive Visualization
""")
# ==========================================================
# FEATURES
# ==========================================================

st.divider()

st.markdown("## 🚀 Key Features")

st.write("""
Guava Disease Detector AI provides several intelligent features to assist users
in detecting guava fruit diseases while offering transparent and interpretable
AI predictions.
""")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### 🔍 AI Disease Detection

Upload a guava fruit image and let the AI model automatically classify it into one of three categories:

- Healthy
- Anthracnose
- Fruit Fly
""")

    st.markdown("""
### 📊 Prediction Confidence

Display confidence scores and probability distributions for all predicted classes.
""")

    st.markdown("""
### 🧠 Explainable AI (Grad-CAM)

Visualize which regions of the image are most influential in the model's prediction using Grad-CAM heatmaps.
""")

with col2:

    st.markdown("""
### 📈 Model Performance Dashboard

Explore the performance comparison of four experimental models, including Accuracy, Precision, Recall, and F1-Score.
""")

    st.markdown("""
### 📚 Disease Information

Learn about each disease, including:

- Description
- Symptoms
- Prevention
- Treatment
""")

    st.markdown("""
### ⚡ Fast Inference

Generate predictions within seconds using a lightweight MobileNetV2 architecture optimized for web deployment.
""")

st.divider()

st.markdown("### ✅ Feature Checklist")

feature_df = pd.DataFrame({
    "Feature": [
        "Image Upload",
        "Automatic Disease Classification",
        "Prediction Confidence",
        "Top Probability Visualization",
        "Grad-CAM Explainable AI",
        "Disease Information",
        "Model Performance Dashboard",
        "Interactive Streamlit Interface"
    ],
    "Status": [
        "✅",
        "✅",
        "✅",
        "✅",
        "✅",
        "✅",
        "✅",
        "✅"
    ]
})

st.dataframe(
    feature_df,
    hide_index=True,
    use_container_width=True
)

st.success("""
This application combines Deep Learning, Explainable AI, and an interactive
web interface to provide an end-to-end solution for guava fruit disease
classification.
""")