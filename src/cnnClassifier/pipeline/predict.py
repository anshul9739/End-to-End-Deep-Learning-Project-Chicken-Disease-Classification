from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


def _find_repo_root(start_file: Path) -> Path:
    """
    Walk upwards from this file until we find a folder that
    looks like the project root (has artifacts/ or config/config.yaml).
    """
    cur = start_file.resolve().parent
    for ancestor in [cur, *cur.parents]:
        if (ancestor / "artifacts").exists() or (ancestor / "config" / "config.yaml").exists():
            return ancestor
    # Fallback: typical .../src/cnnClassifier/pipeline/ -> repo is 3 parents up from src
    return start_file.resolve().parents[3]


def _resolve_model_path() -> Path:
    repo_root = _find_repo_root(Path(__file__))
    primary = repo_root / "artifacts" / "training" / "model.h5"
    fallback = repo_root / "artifacts" / "prepare_base_model" / "base_model_updated.h5"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find a model file.\n"
        f"Tried:\n  - {primary}\n  - {fallback}\n"
        f"Run `python main.py` from the project root to train and create the model."
    )


class PredictionPipeline:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def predict(self) -> List[Dict[str, str]]:
        model_path = _resolve_model_path()
        model = load_model(str(model_path))

        img = load_img(self.filename, target_size=(224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)
        idx = int(np.argmax(preds, axis=1)[0])

        label = "Healthy" if idx == 1 else "Coccidiosis"
        return [{"image": label}]

