from pathlib import Path
import shutil

import tensorflow as tf
import yaml

from src.augmentations import apply_custom_augmentations


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f) or {}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_image(image_uint8: tf.Tensor, ext: str) -> bytes:
    if ext in {".png"}:
        return tf.io.encode_png(image_uint8).numpy()
    return tf.io.encode_jpeg(image_uint8, quality=95).numpy()


def main():
    params = load_params()
    aug = params.get("augment", {})

    input_dir = Path(aug.get("input_dir", "data/raw/skin-moles-benign-vs-malignant-melanoma-isic19"))
    output_dir = Path(aug.get("output_dir", "data/augmented"))
    img_size = int(aug.get("img_size", 224))
    copies_per_image = int(aug.get("copies_per_image", 1))
    include_original = bool(aug.get("include_original", True))
    seed = int(aug.get("seed", 42))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_dir}")
    if copies_per_image < 0:
        raise ValueError("augment.copies_per_image must be >= 0")

    tf.keras.utils.set_random_seed(seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in: {input_dir}")

    written = 0
    for class_dir in class_dirs:
        target_class_dir = output_dir / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in class_dir.rglob("*") if p.is_file() and is_image_file(p)])
        for image_path in image_paths:
            raw = tf.io.read_file(str(image_path))
            image = tf.image.decode_image(raw, channels=3, expand_animations=False)
            image = tf.image.resize(image, [img_size, img_size])
            image = tf.cast(image, tf.float32) / 255.0

            stem = image_path.stem
            ext = image_path.suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png"}:
                ext = ".jpg"

            if include_original:
                original_uint8 = tf.cast(tf.clip_by_value(image, 0.0, 1.0) * 255.0, tf.uint8)
                out_path = target_class_dir / f"{stem}__orig{ext}"
                tf.io.write_file(str(out_path), encode_image(original_uint8, ext))
                written += 1

            for i in range(copies_per_image):
                augmented = apply_custom_augmentations(image)
                augmented_uint8 = tf.cast(tf.clip_by_value(augmented, 0.0, 1.0) * 255.0, tf.uint8)
                out_path = target_class_dir / f"{stem}__aug{i + 1}{ext}"
                tf.io.write_file(str(out_path), encode_image(augmented_uint8, ext))
                written += 1

    print(f"Prepared augmented dataset at {output_dir} with {written} images")


if __name__ == "__main__":
    main()
