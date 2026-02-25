# src/train.py
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
import mlflow

from .utils import configure_mlflow_for_dagshub, set_common_mlflow_tags
from .data import build_datasets  # if you made data.py; otherwise inline dataset code
from .modeling import build_classifier


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f) or {}


def main():
    params = load_params()
    t = params["train"]

    seed = int(t.get("seed", 42))
    tf.keras.utils.set_random_seed(seed)

    dataset_dir = t["dataset_dir"]
    img_size = int(t.get("img_size", 224))
    image_size = (img_size, img_size)
    batch_size = int(t.get("batch_size", 32))
    epochs = int(t.get("epochs", 50))
    lr = float(t.get("lr", 0.01))
    val_split = float(t.get("validation_split", 0.2))
    test_split = float(t.get("testing_split", 0.1))
    base_weights = t.get("base_weights", "imagenet")
    online_augmentation = bool(t.get("online_augmentation", False))

    Path("models").mkdir(parents=True, exist_ok=True)

    using_dagshub = configure_mlflow_for_dagshub()

    train_ds, val_ds, test_ds = build_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=val_split,
        testing_split=test_split,
        seed=seed,
        augment_train=online_augmentation,
    )

    model = build_classifier(image_size=image_size, base_weights=base_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    with mlflow.start_run(run_name=t.get("run_name", None)):
        set_common_mlflow_tags(
            {
                "tracking": "dagshub" if using_dagshub else "local",
                "model": "MobileNetV2",
                "img_size": str(img_size),
            }
        )

        # params
        mlflow.log_params(
            {
                "seed": seed,
                "dataset_dir": dataset_dir,
                "img_size": img_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "validation_split": val_split,
                "testing_split": test_split,
                "base_weights": str(base_weights),
                "online_augmentation": online_augmentation,
            }
        )

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

        test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=1)

        model_path = "models/model.keras"
        model.save(model_path)


        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_auc": float(test_auc),
            "best_val_loss": float(np.min(history.history["val_loss"])),
            "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
            "best_val_auc": float(np.max(history.history["val_auc"])),
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_metrics(metrics)
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
