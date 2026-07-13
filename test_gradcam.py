from utils.predictor import get_model

model = get_model()

print("="*50)

for layer in model.layers:
    print(layer.name)