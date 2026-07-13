import cv2
import numpy as np
import tensorflow as tf

LAST_CONV_LAYER = "out_relu"


def generate_gradcam(model, image_array):
    """
    Generate Grad-CAM heatmap.
    image_array shape = (1,224,224,3)
    """

    grad_model = tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image_array)

        class_index = tf.argmax(predictions[0])

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = tf.maximum(heatmap,0)

    heatmap /= tf.reduce_max(heatmap)+1e-8

    return heatmap.numpy()


def overlay_heatmap(original_image, heatmap, alpha=0.45):

    image = np.array(original_image.convert("RGB"))

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image,
        1-alpha,
        heatmap,
        alpha,
        0
    )

    return overlay