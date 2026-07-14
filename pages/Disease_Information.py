import streamlit as st
from PIL import Image
import glob
import os

from utils.disease_data import disease_information

# =====================
# CSS
# =====================

with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =====================
# Header
# =====================

st.markdown("""
<div class="main-title">
🍈 Disease Information
</div>

<div class="subtitle">
Learn about guava fruit diseases, symptoms, causes, and prevention.
</div>
""", unsafe_allow_html=True)

st.divider()

# =====================
# Disease Cards
# =====================

folder_map = {
    "Anthracnose": "Anthracnose",
    "Fruit Fly": "fruit_fly",
    "Healthy": "healthy_guava"
}

for disease, info in disease_information.items():

    st.subheader(disease)

    # 1. Tampilkan 3 gambar sejajar horizontal
    folder_name = folder_map.get(disease)
    if folder_name:
        search_path = os.path.join("data", "JAMBU_BIJI", "test", folder_name, "*.png")
        img_paths = sorted(glob.glob(search_path))[:3]
        
        if img_paths:
            img_cols = st.columns(3)
            for idx, path in enumerate(img_paths):
                with img_cols[idx]:
                    st.image(path, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Penjelasan disease di bawah gambar
    st.write(info["description"])

    detail_cols = st.columns(3)

    with detail_cols[0]:
        if "symptoms" in info:
            st.markdown("### Symptoms")
            for item in info["symptoms"]:
                st.write("•", item)
        elif "characteristics" in info:
            st.markdown("### Characteristics")
            for item in info["characteristics"]:
                st.write("•", item)

    with detail_cols[1]:
        st.markdown("### Causes")
        if "causes" in info:
            for item in info["causes"]:
                st.write("•", item)
        else:
            st.write("None")

    with detail_cols[2]:
        if "prevention" in info:
            st.markdown("### Prevention")
            for item in info["prevention"]:
                st.write("•", item)
        elif "recommendation" in info:
            st.markdown("### Recommendation")
            for item in info["recommendation"]:
                st.write("•", item)

    st.divider()