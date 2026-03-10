from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "bundle_manifest.txt"
OUTPUT_ZIP = PROJECT_ROOT / "training_bundle.zip"


def iter_manifest_paths(manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel or rel.startswith("#"):
                continue
            yield rel


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest file: {MANIFEST_PATH}")

    included = []

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_str in iter_manifest_paths(MANIFEST_PATH):
            abs_path = PROJECT_ROOT / rel_str

            if not abs_path.exists():
                raise FileNotFoundError(f"Manifest path does not exist: {abs_path}")

            if abs_path.is_file():
                zf.write(abs_path, arcname=rel_str.replace("\\", "/"))
                included.append(rel_str)
            else:
                for file_path in sorted(abs_path.rglob("*")):
                    if file_path.is_file():
                        arcname = file_path.relative_to(PROJECT_ROOT).as_posix()
                        zf.write(file_path, arcname=arcname)
                        included.append(arcname)

    print(f"Bundle created: {OUTPUT_ZIP}")
    print(f"Files included: {len(included)}")


if __name__ == "__main__":
    main()