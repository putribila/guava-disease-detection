# 🍈 Guava Disease Detector AI

An end-to-end Explainable AI (XAI) web application developed to classify guava fruit diseases using deep learning, built with **Streamlit**, **TensorFlow**, and **OpenCV**.

---
## Live Deploy Dashboard 

🔗 **View Interactive Dashboard:**  
[guava-disease-detector.streamlit.app]([https://datastudio.google.com/reporting/e77ffbee-3848-4ce1-9573-7fe47da3f07b](https://guava-disease-detector.streamlit.app/))

---

## 🚀 Features

1. **Guava Verification (Image Validation)**: Uses a pre-trained ImageNet model to verify that the uploaded file contains a guava fruit before running the disease classification model, preventing incorrect predictions for arbitrary uploads (e.g. cars, cats, other fruits).
2. **AI Disease Classification**: Predicts guava fruit condition into one of 3 categories:
   - **Healthy**
   - **Anthracnose**
   - **Fruit Fly**
3. **Explainable AI (Grad-CAM)**: Generates and overlays activation heatmaps to show the exact regions of the guava fruit that the model paid attention to during inference.
4. **Model Performance Dashboard**: Interactive visual comparisons of the four training strategies (Baseline, Weighted Loss, Oversampling, and Combination) utilizing Plotly charts and training history curves.
5. **Disease Encyclopedia**: Informative section detailing characteristics, symptoms, causes, and recommendations/preventions for each guava condition with image galleries.

---

## 🧠 Model Architecture & Experiments

The classification engine is based on **MobileNetV2** (with pre-trained ImageNet weights). We evaluated 4 distinct training configurations to handle class imbalance:

| Model Scenario | Test Accuracy | Loss | Macro F1-Score | Healthy Class F1-Score |
|---|---|---|---|---|
| **Baseline** | 95.55% | 0.1353 | 95.40% | 95.19% |
| **Weighted Loss** | 94.76% | 0.1648 | 94.79% | 95.65% |
| **Oversampling** | 95.55% | 0.1382 | 95.48% | 95.65% |
| **Combination (Oversampling + Weighted)** | **95.81%** | **0.1127** | **95.77%** | **96.26%** |

*The **Combination (Oversampling + Weighted Loss)** model achieved the highest score and is deployed for live inference.*

---

## 📂 Project Structure

```text
guava-disease-detector-ai/
├── app.py                     # Entrypoint & Page Navigation Routing
├── requirements.txt           # Project dependencies
├── assets/
│   ├── style.css              # Custom UI CSS Styling
│   └── disease/               # Sample disease images
├── data/
│   ├── JAMBU_BIJI/            # Guava dataset split (Train, Val, Test)
│   ├── disease_information.json
│   └── experiment_results.csv
├── notebooks/
│   └── guava_disease_experiment.py
├── pages/
│   ├── About_Project.py       # Overview of architecture & tech stack
│   ├── Disease_Detection.py   # Prediction interface & Grad-CAM visualizer
│   ├── Disease_Information.py # Symptoms, causes, & prevention guides
│   └── Model_Performance.py   # Performance dashboard & training curves
└── utils/
    ├── disease_data.py
    ├── disease_info.py
    ├── gradcam.py             # Grad-CAM heatmap generation & overlays
    ├── predictor.py           # Inference engine & image validation
    └── preprocessing.py       # Image resizing and normalization
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/guava-disease-detector-ai.git
   cd guava-disease-detector-ai
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```
