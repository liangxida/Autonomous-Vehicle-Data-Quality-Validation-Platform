from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "sample"
REPORT_DIR = PROJECT_ROOT / "reports"

METADATA_PATH = DATA_DIR / "metadata.csv"
ANNOTATION_QUALITY_REPORT_PATH = REPORT_DIR / "annotation_quality_report.csv"
ANNOTATION_QUALITY_SUMMARY_PATH = REPORT_DIR / "annotation_quality_summary.json"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

APPROVED_CLASSES = {
    "vehicle",
    "pedestrian",
    "cyclist",
    "traffic_light",
    "traffic_sign",
}


REQUIRED_ANNOTATION_COLUMNS = [
    "frame_id",
    "class",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_columns = ["frame_id", "annotation_file"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]

    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")

    return metadata


def validate_single_annotation_file(frame_id: str, annotation_relative_path: str) -> dict:
    annotation_path = DATA_DIR / str(annotation_relative_path)

    result = {
        "frame_id": frame_id,
        "annotation_file": annotation_relative_path,
        "annotation_file_exists": annotation_path.exists(),
        "annotation_file_readable": False,
        "object_count": 0,
        "invalid_class_count": 0,
        "invalid_bbox_count": 0,
        "frame_id_mismatch_count": 0,
        "annotation_quality_passed": False,
        "failure_reasons": "",
    }

    if not annotation_path.exists():
        result["failure_reasons"] = "missing_annotation_file"
        return result

    try:
        annotations = pd.read_csv(annotation_path)
        result["annotation_file_readable"] = True
    except Exception:
        result["failure_reasons"] = "unreadable_annotation_file"
        return result

    missing_columns = [
        col for col in REQUIRED_ANNOTATION_COLUMNS
        if col not in annotations.columns
    ]

    if missing_columns:
        result["failure_reasons"] = "missing_required_annotation_columns"
        return result

    # Empty annotation file with only header is allowed.
    if annotations.empty:
        result["object_count"] = 0
        result["annotation_quality_passed"] = True
        result["failure_reasons"] = "passed_empty_frame"
        return result

    result["object_count"] = len(annotations)

    invalid_class_count = 0
    invalid_bbox_count = 0
    frame_id_mismatch_count = 0

    for _, row in annotations.iterrows():
        label = str(row["class"])

        if label not in APPROVED_CLASSES:
            invalid_class_count += 1

        if str(row["frame_id"]) != str(frame_id):
            frame_id_mismatch_count += 1

        try:
            x_min = float(row["x_min"])
            y_min = float(row["y_min"])
            x_max = float(row["x_max"])
            y_max = float(row["y_max"])
        except Exception:
            invalid_bbox_count += 1
            continue

        bbox_is_valid = (
            x_min >= 0
            and y_min >= 0
            and x_max <= IMAGE_WIDTH
            and y_max <= IMAGE_HEIGHT
            and x_min < x_max
            and y_min < y_max
        )

        if not bbox_is_valid:
            invalid_bbox_count += 1

    result["invalid_class_count"] = invalid_class_count
    result["invalid_bbox_count"] = invalid_bbox_count
    result["frame_id_mismatch_count"] = frame_id_mismatch_count

    reasons = []

    if invalid_class_count > 0:
        reasons.append("invalid_class_label")

    if invalid_bbox_count > 0:
        reasons.append("invalid_bounding_box")

    if frame_id_mismatch_count > 0:
        reasons.append("frame_id_mismatch")

    if reasons:
        result["annotation_quality_passed"] = False
        result["failure_reasons"] = ";".join(reasons)
    else:
        result["annotation_quality_passed"] = True
        result["failure_reasons"] = "passed"

    return result


def validate_annotations(metadata: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in metadata.iterrows():
        frame_id = row["frame_id"]
        annotation_file = row["annotation_file"]

        result = validate_single_annotation_file(frame_id, annotation_file)
        results.append(result)

    return pd.DataFrame(results)


def summarize_annotation_quality(report: pd.DataFrame) -> dict:
    total_records = len(report)

    passed_count = int(report["annotation_quality_passed"].sum())
    failed_count = total_records - passed_count

    summary = {
        "total_annotation_records": total_records,
        "annotation_quality_passed_count": passed_count,
        "annotation_quality_failed_count": failed_count,
        "annotation_quality_pass_rate": round(passed_count / total_records, 4) if total_records else 0,
        "missing_annotation_file_count": int((~report["annotation_file_exists"]).sum()),
        "unreadable_annotation_file_count": int(((~report["annotation_file_readable"]) & report["annotation_file_exists"]).sum()),
        "total_labeled_objects": int(report["object_count"].sum()),
        "invalid_class_count": int(report["invalid_class_count"].sum()),
        "invalid_bbox_count": int(report["invalid_bbox_count"].sum()),
        "frame_id_mismatch_count": int(report["frame_id_mismatch_count"].sum()),
    }

    return summary


def run_annotation_validation() -> tuple[pd.DataFrame, dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(METADATA_PATH)
    report = validate_annotations(metadata)
    summary = summarize_annotation_quality(report)

    report.to_csv(ANNOTATION_QUALITY_REPORT_PATH, index=False)

    with ANNOTATION_QUALITY_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Annotation quality validation finished.")
    print(f"Report saved to: {ANNOTATION_QUALITY_REPORT_PATH}")
    print(f"Summary saved to: {ANNOTATION_QUALITY_SUMMARY_PATH}")

    return report, summary


if __name__ == "__main__":
    run_annotation_validation()