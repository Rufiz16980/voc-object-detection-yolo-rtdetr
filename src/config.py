from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Core directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
CVAT_EXPORTS_DIR = DATA_DIR / "cvat_exports"

IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
VAL_IMAGES_DIR = IMAGES_DIR / "val"
TEST_IMAGES_DIR = IMAGES_DIR / "test"

TRAIN_LABELS_DIR = LABELS_DIR / "train"
VAL_LABELS_DIR = LABELS_DIR / "val"
TEST_LABELS_DIR = LABELS_DIR / "test"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# VOC dataset paths
VOC_ROOT = RAW_DIR / "VOCdevkit" / "VOC2012"
VOC_IMAGES_DIR = VOC_ROOT / "JPEGImages"
VOC_ANNOTATIONS_DIR = VOC_ROOT / "Annotations"
VOC_IMAGESETS_DIR = VOC_ROOT / "ImageSets" / "Main"

# Reproducibility
SEED = 42

# Assignment target classes
TARGET_CLASSES = ["car", "bus", "bicycle", "motorbike"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}

# Manifest / artifact paths
ANNOTATION_MANIFEST_PATH = INTERIM_DIR / "annotation_manifest.csv"
SELECTED_FOR_CVAT_PATH = INTERIM_DIR / "selected_for_cvat.csv"
CVAT_UPLOAD_DIR = INTERIM_DIR / "cvat_upload_images"
PHASE1_STATE_PATH = INTERIM_DIR / "phase1_state.json"
CLASS_COUNTS_PATH = TABLES_DIR / "class_counts.csv"


def get_required_directories():
    """Return all directories that must exist for Phase 1."""
    return [
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        CVAT_EXPORTS_DIR,
        IMAGES_DIR,
        LABELS_DIR,
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TEST_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_LABELS_DIR,
        TEST_LABELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        CVAT_UPLOAD_DIR,
    ]