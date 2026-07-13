import streamlit as st

def show_home():
    st.title("🍈 Guava Disease Detector AI")

    st.markdown("""
    ### Detect Guava Fruit Diseases Using Deep Learning

    Welcome to **Guava Disease Detector AI**, a web application developed to classify guava fruit diseases using **Transfer Learning (MobileNetV2)**.

    This project demonstrates the implementation of Computer Vision, Deep Learning, and Explainable AI for disease detection.
    """)

    st.divider()

    st.header("🚀 Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🧠 **Image Classification**\n\nPredict guava fruit diseases from uploaded images.")

        st.info("📊 **Model Performance**\n\nCompare the results of four different training experiments.")

    with col2:
        st.info("🔬 **Explainable AI**\n\nVisualize the model's focus using Grad-CAM.")

        st.info("📚 **Disease Information**\n\nLearn about symptoms, causes, and treatments.")

    st.divider()

    st.header("📖 About Project")

    st.write("""
    This application was developed as part of an Artificial Intelligence portfolio project.

    The classification model is based on **MobileNetV2 Transfer Learning** and evaluates four different strategies for handling class imbalance:

    - Baseline
    - Weighted Loss
    - Oversampling
    - Combination (Oversampling + Weighted Loss)
    """)

    st.divider()

    st.caption("Developed by Putri Nabilla")

# Page Config
st.set_page_config(
    page_title="Guava Disease Detector AI",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Define Pages
home_page = st.Page(show_home, title="Home", default=True)
about_page = st.Page("pages/About_Project.py", title="About Project")
detection_page = st.Page("pages/Disease_Detection.py", title="Disease Detection")
info_page = st.Page("pages/Disease_Information.py", title="Disease Information")
performance_page = st.Page("pages/Model_Performance.py", title="Model Performance")

# Navigation Routing
pg = st.navigation([home_page, about_page, detection_page, info_page, performance_page])
pg.run()