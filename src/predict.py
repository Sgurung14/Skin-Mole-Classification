# src/predict.py
import argparse
import json

import tensorflow as tf

from src.modeling import build_classifier, preprocess_input_array


def resolve_model_load_mode(model_path: str, mode: str) -> str:
    value = (mode or "auto").strip().lower()
    if value not in {"auto", "weights", "full"}:
        raise ValueError("--load-mode must be one of: auto, weights, full")
    if value != "auto":
        return value

    path_lower = model_path.lower()
    if path_lower.endswith(".keras"):
        return "full"
    if path_lower.endswith(".weights.h5"):
        return "weights"
    if path_lower.endswith(".h5"):
        return "full"
    return "weights"


def load_and_preprocess(image_path: str, img_size: int, load_mode: str):
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)

    if load_mode == "full":
        # Full saved models may include preprocessing inside the model graph.
        return img

    return tf.convert_to_tensor(preprocess_input_array(img.numpy()), dtype=tf.float32)


def load_model(model_path: str, img_size: int, load_mode: str):
    if load_mode == "full":
        return tf.keras.models.load_model(model_path, compile=False)

    model = build_classifier(
        image_size=(img_size, img_size),
        base_weights=None,
    )
    model.load_weights(model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default="models/model.weights.h5", help="Path to saved model or weights")
    parser.add_argument("--load-mode", default="auto", choices=["auto", "weights", "full"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    load_mode = resolve_model_load_mode(args.model, args.load_mode)
    model = load_model(args.model, args.img_size, load_mode)
    x = load_and_preprocess(args.image, args.img_size, load_mode)
    prob = float(model.predict(x, verbose=0).reshape(-1)[0])
    pred = int(prob >= args.threshold)

    result = {
        "prob_malignant": prob,
        "pred_label": "malignant" if pred == 1 else "benign",
        "threshold": args.threshold,
        "model_load_mode": load_mode,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
