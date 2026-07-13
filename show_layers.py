import tensorflow as tf

model = tf.keras.models.load_model(
    "model/Kombinasi_(Over_Weight).h5",
    compile=False
)

print("="*60)
print("Model Layers")
print("="*60)

for i, layer in enumerate(model.layers):
    print(f"{i:2d} - {layer.name} ({layer.__class__.__name__})")