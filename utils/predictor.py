import json
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/Kombinasi_(Over_Weight).h5"
CLASS_PATH = "model/class_names.json"

# Load model hanya sekali
model = tf.keras.models.load_model(MODEL_PATH)

# Load nama kelas
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)


def predict(image):
    """
    Predict image and return:
    class,
    confidence,
    all probabilities
    """

    prediction = model.predict(image, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = prediction[predicted_index]

    return predicted_class, confidence, prediction

def get_model():
    return model

imagenet_model = None

def is_guava(image_pil):
    """
    Validate if the uploaded image is a guava fruit/plant/visual lookalike using MobileNetV2 ImageNet.
    """
    global imagenet_model
    if imagenet_model is None:
        imagenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    
    # Preprocess image for MobileNetV2 ImageNet
    img = image_pil.convert("RGB").resize((224, 224))
    x = np.array(img).astype("float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    
    preds = imagenet_model.predict(x, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
    
    print("ImageNet Top Predictions for uploaded image:")
    for item in decoded:
        print(f"  - {item[1]}: {item[2]*100:.2f}%")
    
    # Strict set of keywords matching guava visual features (custard apple, pear, green apple, fig, lemon, pomegranate) and dataset anomalies (mortar, tennis ball, guacamole, croquet ball)
    allowed_keywords = {
        'custard_apple', 'fig', 'pear', 'granny_smith', 'lemon', 'pomegranate', 'guava',
        'mortar', 'tennis_ball', 'guacamole', 'croquet_ball'
    }
    
    for item in decoded:
        pred_class = item[1].lower()
        if any(kw in pred_class for kw in allowed_keywords):
            return True
            
    return False
