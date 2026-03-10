from pathlib import Path
from typing import Dict, List

import pandas as pd


def find_latest_cvat_export(cvat_exports_dir: Path) -> Path:
    """
    Find the most recent CVAT export directory.

    Args:
        cvat_exports_dir: Directory containing CVAT export folders.

    Returns:
        Path to the most recent export folder.
    """
    candidates = [p for p in cvat_exports_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No CVAT export folders found in: {cvat_exports_dir}")

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_cvat_class_names(export_dir: Path) -> List[str]:
    """
    Load class names from obj.names in a CVAT YOLO export.

    Args:
        export_dir: CVAT export directory.

    Returns:
        List of class names.
    """
    names_path = export_dir / "obj.names"
    if not names_path.exists():
        raise FileNotFoundError(f"Missing obj.names in export: {export_dir}")

    with open(names_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    return class_names


def build_export_label_index(export_dir: Path) -> Dict[str, Path]:
    """
    Build a mapping from image stem to YOLO label file path.

    Args:
        export_dir: CVAT export directory.

    Returns:
        Dictionary mapping image stem -> label path.
    """
    labels_dir = export_dir / "obj_train_data"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing obj_train_data directory in export: {export_dir}")

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found in: {labels_dir}")

    return {label_path.stem: label_path for label_path in label_files}


def attach_cvat_labels_to_selection(
    selected_df: pd.DataFrame,
    export_dir: Path,
) -> pd.DataFrame:
    """
    Attach CVAT-exported label paths to the selected image DataFrame.

    Args:
        selected_df: DataFrame of selected images.
        export_dir: CVAT export directory.

    Returns:
        DataFrame with added label_path column.

    Raises:
        ValueError: If some selected images are missing corresponding label files.
    """
    label_index = build_export_label_index(export_dir)
    df = selected_df.copy()

    df["label_path"] = df["image_id"].map(lambda image_id: str(label_index.get(image_id, "")))

    missing = df[df["label_path"] == ""]
    if not missing.empty:
        missing_ids = missing["image_id"].tolist()
        raise ValueError(
            "Some selected images do not have exported CVAT labels. "
            f"Missing image_ids: {missing_ids[:10]}"
            + (" ..." if len(missing_ids) > 10 else "")
        )

    return df