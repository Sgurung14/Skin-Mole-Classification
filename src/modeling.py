import tensorflow as tf


def build_classifier(image_size, base_weights="imagenet"):
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=base_weights,
        input_shape=image_size + (3,),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=image_size + (3,))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="MobileNetV2_SkinMole")


def preprocess_input_array(arr):
    """Preprocess an RGB image array (0..255) to match the training pipeline."""
    x = tf.convert_to_tensor(arr, dtype=tf.float32)
    x = x / 255.0
    return x.numpy()
