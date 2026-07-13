from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)


def preprocess_image(image):
    """
    Preprocess image before prediction.
    """

    # Pastikan gambar RGB
    image = image.convert("RGB")

    # Resize sesuai ukuran model
    image = image.resize(IMG_SIZE)

    # Convert ke numpy array
    image = np.array(image)

    # Normalisasi
    image = image.astype("float32") / 255.0

    # Tambahkan batch dimension
    image = np.expand_dims(image, axis=0)

    return image