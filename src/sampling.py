from pathlib import Path

import pandas as pd


def _take_unique_rows(source_df: pd.DataFrame, selected_ids: set[str], n: int) -> list[dict]:
    """
    Take up to n unique rows from source_df, skipping already selected image_ids.

    Args:
        source_df: Candidate DataFrame.
        selected_ids: Set of already selected image IDs.
        n: Maximum number of rows to take.

    Returns:
        List of row dictionaries.
    """
    rows = []
    for _, row in source_df.iterrows():
        image_id = row["image_id"]
        if image_id in selected_ids:
            continue
        rows.append(row.to_dict())
        selected_ids.add(image_id)
        if len(rows) >= n:
            break
    return rows


def select_images_for_cvat(
    manifest_df: pd.DataFrame,
    target_num_images: int = 150,
) -> pd.DataFrame:
    """
    Select a more balanced subset of images for manual annotation in CVAT.

    Strategy:
    1. Reserve quota for bicycle-rich images
    2. Reserve quota for motorbike-rich images
    3. Reserve quota for bus-rich images
    4. Fill remaining slots with globally strong mixed images

    This keeps the subset from collapsing into one dominant class.

    Args:
        manifest_df: Candidate manifest DataFrame.
        target_num_images: Number of images to select.

    Returns:
        DataFrame containing the selected images.
    """
    if manifest_df.empty:
        return manifest_df.copy()

    required_columns = {
        "image_id",
        "filename",
        "image_path",
        "xml_path",
        "width",
        "height",
        "classes_present",
        "car_count",
        "bus_count",
        "bicycle_count",
        "motorbike_count",
        "total_target_objects",
    }

    missing_columns = required_columns - set(manifest_df.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

    df = manifest_df.copy()

    # Helper features
    df["has_car"] = (df["car_count"] > 0).astype(int)
    df["has_bus"] = (df["bus_count"] > 0).astype(int)
    df["has_bicycle"] = (df["bicycle_count"] > 0).astype(int)
    df["has_motorbike"] = (df["motorbike_count"] > 0).astype(int)

    df["num_target_classes_present"] = (
        df["has_car"] + df["has_bus"] + df["has_bicycle"] + df["has_motorbike"]
    )

    # Mixed-scene score for the final fill stage
    df["mixed_score"] = (
        1.0 * df["car_count"]
        + 2.0 * df["bus_count"]
        + 2.0 * df["bicycle_count"]
        + 2.0 * df["motorbike_count"]
        + 4.0 * df["num_target_classes_present"]
        + 0.25 * df["total_target_objects"]
    )

    # Class-focused rankings
    bicycle_df = df[df["bicycle_count"] > 0].sort_values(
        by=["bicycle_count", "num_target_classes_present", "bus_count", "motorbike_count", "car_count", "total_target_objects", "image_id"],
        ascending=[False, False, False, False, False, False, True],
    )

    motorbike_df = df[df["motorbike_count"] > 0].sort_values(
        by=["motorbike_count", "num_target_classes_present", "bus_count", "bicycle_count", "car_count", "total_target_objects", "image_id"],
        ascending=[False, False, False, False, False, False, True],
    )

    bus_df = df[df["bus_count"] > 0].sort_values(
        by=["bus_count", "num_target_classes_present", "bicycle_count", "motorbike_count", "car_count", "total_target_objects", "image_id"],
        ascending=[False, False, False, False, False, False, True],
    )

    mixed_df = df.sort_values(
        by=["mixed_score", "num_target_classes_present", "bicycle_count", "motorbike_count", "bus_count", "car_count", "total_target_objects", "image_id"],
        ascending=[False, False, False, False, False, False, False, True],
    )

    selected_ids = set()
    selected_rows = []

    # Quotas for 150 images
    bicycle_quota = 40
    motorbike_quota = 40
    bus_quota = 35

    selected_rows.extend(_take_unique_rows(bicycle_df, selected_ids, bicycle_quota))
    selected_rows.extend(_take_unique_rows(motorbike_df, selected_ids, motorbike_quota))
    selected_rows.extend(_take_unique_rows(bus_df, selected_ids, bus_quota))

    remaining = target_num_images - len(selected_rows)
    if remaining > 0:
        selected_rows.extend(_take_unique_rows(mixed_df, selected_ids, remaining))

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    # Truncate just in case
    selected_df = selected_df.head(target_num_images).copy()

    helper_columns = [
        "has_car",
        "has_bus",
        "has_bicycle",
        "has_motorbike",
        "num_target_classes_present",
        "mixed_score",
    ]
    selected_df = selected_df.drop(
        columns=[col for col in helper_columns if col in selected_df.columns],
        errors="ignore",
    )

    selected_df["selected_for_cvat"] = True
    return selected_df


def copy_selected_images_to_cvat_folder(
    selected_df: pd.DataFrame,
    cvat_upload_dir: Path,
) -> None:
    """
    Copy selected images into a dedicated CVAT upload folder.

    Args:
        selected_df: DataFrame of selected images.
        cvat_upload_dir: Destination folder for CVAT upload images.
    """
    cvat_upload_dir.mkdir(parents=True, exist_ok=True)

    for _, row in selected_df.iterrows():
        src_path = Path(row["image_path"])
        dst_path = cvat_upload_dir / src_path.name

        if not src_path.exists():
            raise FileNotFoundError(f"Selected image does not exist: {src_path}")

        if not dst_path.exists():
            dst_path.write_bytes(src_path.read_bytes())