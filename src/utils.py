# src/utils.py
import os
import subprocess
from typing import Optional

import mlflow


def get_git_sha() -> Optional[str]:
    """Return current git commit SHA or None if unavailable."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return sha.decode().strip()
    except Exception:
        return None


def configure_mlflow_for_dagshub() -> bool:
    """
    Configure MLflow to log to DagsHub if env vars are present.
    Required env vars:
      - DAGSHUB_TOKEN
      - DAGSHUB_OWNER
      - DAGSHUB_REPO
    """
    token = os.getenv("DAGSHUB_TOKEN")
    owner = os.getenv("DAGSHUB_OWNER")
    repo = os.getenv("DAGSHUB_REPO")

    if not (token and owner and repo):
        # still okay (logs locally)
        mlflow.set_experiment("skin-mole-classification")
        return False

    mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{repo}.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_experiment("skin-mole-classification")
    return True


def set_common_mlflow_tags(extra: Optional[dict] = None) -> None:
    sha = get_git_sha()
    if sha:
        mlflow.set_tag("git_sha", sha)

    runner = "kaggle" if os.getenv("KAGGLE_KERNEL_RUN_TYPE") else "local"
    mlflow.set_tag("runner", runner)

    if extra:
        for k, v in extra.items():
            mlflow.set_tag(k, v)
