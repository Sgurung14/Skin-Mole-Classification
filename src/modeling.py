import tensorflow as tf


SUPPORTED_BACKBONES = {"efficientnetb0", "resnet101"}


def _normalize_backbone(backbone: str | None) -> str:
    name = (backbone or "efficientnetb0").strip().lower()
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Supported: {sorted(SUPPORTED_BACKBONES)}"
        )
    return name


def build_classifier(image_size, base_weights="imagenet", backbone="efficientnetb0"):
    backbone_name = _normalize_backbone(backbone)

    if backbone_name == "resnet101":
        base = tf.keras.applications.ResNet101(
            include_top=False,
            weights=base_weights,
            input_shape=image_size + (3,),
        )
        base.trainable = False

        inputs = tf.keras.Input(shape=image_size + (3,))
        x = tf.keras.applications.resnet.preprocess_input(inputs)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, outputs, name="ResNet101_SkinMole")

    base = tf.keras.applications.EfficientNetB0(
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
    return tf.keras.Model(inputs, outputs)


def preprocess_input_array(arr, backbone="efficientnetb0"):
    """Preprocess an RGB image array (0..255) for the selected backbone."""
    backbone_name = _normalize_backbone(backbone)
    x = tf.convert_to_tensor(arr, dtype=tf.float32)

    if backbone_name == "resnet101":
        # ResNet preprocessing is applied inside the ResNet101 model graph.
        return x.numpy()

    x = x / 255.0
    return x.numpy()
