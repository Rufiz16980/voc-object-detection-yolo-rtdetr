from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree


def parse_voc_xml(xml_path: Path) -> dict[str, Any]:
    """
    Parse a Pascal VOC XML annotation file.

    Args:
        xml_path: Path to a VOC XML file.

    Returns:
        Dictionary containing image metadata and object annotations.
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", default="0"))
        truncated = int(obj.findtext("truncated", default="0"))
        occluded = 0  # VOC 2012 standard annotation does not explicitly provide occluded
        bbox = obj.find("bndbox")

        xmin = int(float(bbox.findtext("xmin")))
        ymin = int(float(bbox.findtext("ymin")))
        xmax = int(float(bbox.findtext("xmax")))
        ymax = int(float(bbox.findtext("ymax")))

        objects.append(
            {
                "class_name": name,
                "difficult": difficult,
                "truncated": truncated,
                "occluded": occluded,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return {
        "image_id": xml_path.stem,
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def build_candidate_manifest(
    annotations_dir: Path,
    images_dir: Path,
    target_classes: list[str],
) -> pd.DataFrame:
    """
    Build a manifest of images containing at least one target class.

    Args:
        annotations_dir: Directory containing VOC XML annotation files.
        images_dir: Directory containing VOC JPEG images.
        target_classes: List of target class names.

    Returns:
        DataFrame with one row per candidate image.
    """
    rows = []

    xml_files = sorted(annotations_dir.glob("*.xml"))
    for xml_path in xml_files:
        ann = parse_voc_xml(xml_path)

        class_counts = {class_name: 0 for class_name in target_classes}
        target_objects = []

        for obj in ann["objects"]:
            if obj["class_name"] in target_classes:
                class_counts[obj["class_name"]] += 1
                target_objects.append(obj)

        total_target_objects = sum(class_counts.values())
        if total_target_objects == 0:
            continue

        image_path = images_dir / ann["filename"]
        present_classes = [cls for cls, count in class_counts.items() if count > 0]

        rows.append(
            {
                "image_id": ann["image_id"],
                "filename": ann["filename"],
                "image_path": str(image_path),
                "xml_path": str(xml_path),
                "width": ann["width"],
                "height": ann["height"],
                "classes_present": ",".join(present_classes),
                "car_count": class_counts["car"],
                "bus_count": class_counts["bus"],
                "bicycle_count": class_counts["bicycle"],
                "motorbike_count": class_counts["motorbike"],
                "total_target_objects": total_target_objects,
            }
        )

    manifest_df = pd.DataFrame(rows)

    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(
            by=["total_target_objects", "bus_count", "motorbike_count", "bicycle_count", "car_count"],
            ascending=False,
        ).reset_index(drop=True)

    return manifest_df


def summarize_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple class count summary from the candidate manifest.

    Args:
        manifest_df: Candidate manifest DataFrame.

    Returns:
        Summary DataFrame with total object counts per class.
    """
    if manifest_df.empty:
        return pd.DataFrame(
            {
                "class_name": ["car", "bus", "bicycle", "motorbike"],
                "object_count": [0, 0, 0, 0],
            }
        )

    summary = pd.DataFrame(
        {
            "class_name": ["car", "bus", "bicycle", "motorbike"],
            "object_count": [
                int(manifest_df["car_count"].sum()),
                int(manifest_df["bus_count"].sum()),
                int(manifest_df["bicycle_count"].sum()),
                int(manifest_df["motorbike_count"].sum()),
            ],
        }
    )
    return summary