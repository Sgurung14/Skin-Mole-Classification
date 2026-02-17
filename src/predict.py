# src/predict.py
import argparse
import json

import tensorflow as tf

from src.modeling import build_classifier


def load_and_preprocess(image_path: str, img_size: int):
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default="models/model.weights.h5", help="Path to saved model weights")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = build_classifier(image_size=(args.img_size, args.img_size), base_weights=None)
    model.load_weights(args.model)
    x = load_and_preprocess(args.image, args.img_size)
    prob = float(model.predict(x, verbose=0).reshape(-1)[0])
    pred = int(prob >= args.threshold)

    result = {
        "prob_malignant": prob,
        "pred_label": "malignant" if pred == 1 else "benign",
        "threshold": args.threshold,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#run command : python -m src.predict --image path/to.jpg
