import json
import os
import pickle
from pathlib import Path

import yaml

def main():
    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f) or {}
    train_params = params.get("train", {})
    lr = float(train_params.get("lr", 0.001))
    epochs = int(train_params.get("epochs", 1))

    # Make sure output dirs exist
    Path("models").mkdir(parents=True, exist_ok=True)

    # TODO: Replace this with your real training code.
    # This placeholder "model" proves the pipeline works end-to-end.
    model = {
        "type": "placeholder",
        "lr": lr,
        "epochs": epochs,
    }

    # Save model artifact
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Write metrics
    metrics = {
        "lr": lr,
        "epochs": epochs,
        "status": "ok",
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
