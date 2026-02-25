# src/evaluate.py
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
import mlflow
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

import matplotlib.pyplot as plt

from src.utils import configure_mlflow_for_dagshub, describe_mlflow_tracking, set_common_mlflow_tags
from src.data import build_datasets  # reuse same splitting and preprocessing


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f) or {}


def plot_confusion_matrix(cm, class_names, out_path: str):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, out_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return float(roc_auc)


def main():
    params = load_params()
    t = params["train"]

    seed = int(t.get("seed", 42))
    tf.keras.utils.set_random_seed(seed)

    dataset_dir = t["dataset_dir"]
    img_size = int(t.get("img_size", 224))
    image_size = (img_size, img_size)
    batch_size = int(t.get("batch_size", 32))
    val_split = float(t.get("validation_split", 0.2))
    test_split = float(t.get("testing_split", 0.1))

    model_path = "models/model.keras"
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    using_dagshub = configure_mlflow_for_dagshub()
    describe_mlflow_tracking()

    # Build datasets with identical split logic
    _, _, test_ds = build_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=val_split,
        testing_split=test_split,
        seed=seed,
    )

    model = tf.keras.models.load_model(model_path, compile=False)

    # Collect predictions
    y_true = []
    y_prob = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().reshape(-1).astype(int).tolist())

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    class_names = ["benign", "malignant"]

    cm_path = str(reports_dir / "confusion_matrix.png")
    roc_path = str(reports_dir / "roc_curve.png")

    plot_confusion_matrix(cm, class_names, cm_path)
    roc_auc = plot_roc_curve(y_true, y_prob, roc_path)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    eval_metrics = {
        "eval_roc_auc": float(roc_auc),
        "eval_tp": int(cm[1, 1]),
        "eval_tn": int(cm[0, 0]),
        "eval_fp": int(cm[0, 1]),
        "eval_fn": int(cm[1, 0]),
        "eval_precision_malignant": float(report["malignant"]["precision"]),
        "eval_recall_malignant": float(report["malignant"]["recall"]),
        "eval_f1_malignant": float(report["malignant"]["f1-score"]),
    }

    with open("eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # Log to MLflow (attach to a new run, or you can link by tags)
    with mlflow.start_run(run_name="evaluate"):
        set_common_mlflow_tags({"tracking": "dagshub" if using_dagshub else "local", "stage": "evaluate"})
        mlflow.log_metrics(eval_metrics)
        mlflow.log_artifact("eval_metrics.json")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)


if __name__ == "__main__":
    main()
