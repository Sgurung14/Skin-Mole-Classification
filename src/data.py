import tensorflow as tf
from .augmentations import apply_custom_augmentations

def build_datasets(
    dataset_dir,
    image_size,
    batch_size,
    validation_split,
    testing_split,
    seed,
):
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    full_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=image_size,
        batch_size=None,
        seed=seed,
        shuffle=True,
        label_mode="binary",
    )

    total_size = full_dataset.cardinality().numpy()
    test_size = int(total_size * testing_split)
    val_size = int(total_size * validation_split)

    test_ds = full_dataset.take(test_size)
    remaining = full_dataset.skip(test_size)
    val_ds = remaining.take(val_size)
    train_ds = remaining.skip(val_size)

    def preprocess(image, label):
        image = tf.image.resize(image, image_size)
        image = normalization_layer(image)
        return image, label

    def preprocess_and_augment(image, label):
        image = tf.image.resize(image, image_size)
        image = normalization_layer(image)
        image = apply_custom_augmentations(image)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess_and_augment, num_parallel_calls=AUTOTUNE)\
                       .batch(batch_size)\
                       .prefetch(AUTOTUNE)

    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)\
                   .batch(batch_size)\
                   .prefetch(AUTOTUNE)

    test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)\
                     .batch(batch_size)\
                     .prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds
