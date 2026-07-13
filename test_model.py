import tensorflow as tf

print("TensorFlow :", tf.__version__)
print("Keras      :", tf.keras.__version__)

MODEL_PATH = "model/Kombinasi_(Over_Weight).h5"

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("\n[OK] Model berhasil dimuat!")
model.summary()