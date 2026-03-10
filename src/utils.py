import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Iterable[Path]) -> None:
    """
    Create multiple directories if they do not already exist.

    Args:
        paths: Iterable of directory paths.
    """
    for path in paths:
        ensure_dir(path)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Args:
        path: Input CSV path.

    Returns:
        Loaded DataFrame.
    """
    return pd.read_csv(path)


def save_json(data: dict[str, Any], path: Path) -> None:
    """
    Save a dictionary as JSON.

    Args:
        data: Dictionary to save.
        path: Output JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file into a dictionary.

    Args:
        path: Input JSON path.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_exists_and_nonempty(path: Path) -> bool:
    """
    Check whether a file exists and is non-empty.

    Args:
        path: File path.

    Returns:
        True if file exists and size > 0, else False.
    """
    return path.exists() and path.is_file() and path.stat().st_size > 0