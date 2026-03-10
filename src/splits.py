import random
import shutil
from pathlib import Path

import pandas as pd


def assign_image_split(index: int, total: int) -> str:
    """
    Assign split name based on index position.

    Args:
        index: Row index in shuffled order.
        total: Total number of images.

    Returns:
        One of: train, val, test
    """
    train_cut = int(total * 0.70)
    val_cut = int(total * 0.85)

    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def make_deterministic_splits(selected_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Shuffle deterministically and assign train/val/test splits.

    Args:
        selected_df: DataFrame with selected annotated images.
        seed: Random seed.

    Returns:
        DataFrame with split column.
    """
    df = selected_df.sample(frac=1, random_state=seed).reset_index(drop=True).copy()
    df["split"] = [assign_image_split(i, len(df)) for i in range(len(df))]
    return df


def copy_dataset_to_split_dirs(
    split_df: pd.DataFrame,
    train_images_dir: Path,
    val_images_dir: Path,
    test_images_dir: Path,
    train_labels_dir: Path,
    val_labels_dir: Path,
    test_labels_dir: Path,
) -> None:
    """
    Copy images and labels into final split directories.

    Args:
        split_df: DataFrame with image_path, label_path, split columns.
        train_images_dir: Train images directory.
        val_images_dir: Validation images directory.
        test_images_dir: Test images directory.
        train_labels_dir: Train labels directory.
        val_labels_dir: Validation labels directory.
        test_labels_dir: Test labels directory.
    """
    split_to_dirs = {
        "train": (train_images_dir, train_labels_dir),
        "val": (val_images_dir, val_labels_dir),
        "test": (test_images_dir, test_labels_dir),
    }

    for _, row in split_df.iterrows():
        split_name = row["split"]
        image_src = Path(row["image_path"])
        label_src = Path(row["label_path"])

        image_dst_dir, label_dst_dir = split_to_dirs[split_name]
        image_dst = image_dst_dir / image_src.name
        label_dst = label_dst_dir / label_src.name

        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)