from __future__ import annotations
import base64
import json
from pathlib import Path
from typing import Any, Dict
import yaml


def read_yaml(path_to_yaml: str | Path) -> Dict[str, Any]:
    path = Path(path_to_yaml)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_directories(paths: list[Path | str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def decodeImage(imgstring: str, fileName: str) -> None:
    """Decode base64 image string to a file path."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)


def get_size(path: str | Path) -> str:
    """
    Return approximate file size as a human string like '~ 11345 KB'.
    Matches the logging style your pipeline expects.
    """
    p = Path(path)
    if not p.exists():
        return "~ 0 KB"
    if p.is_dir():
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        total = p.stat().st_size
    return f"~ {round(total / 1024)} KB"


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """
    Save a dictionary to a JSON file with pretty formatting.
    Creates parent directories if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


