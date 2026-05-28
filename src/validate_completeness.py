from pathlib import Path
import json
import pandas as pd

PORJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PORJECT_ROOT / "data"/"sample"
REPORT_DIR = PORJECT_ROOT / "reports"

METADATA_PATH = DATA_DIR / "metadata.csv"
COMPLETENESS_REPORT_PATH = REPORT_DIR / "completeness_report.csv"
COMPLETENESS_SUMMARY_PATH = REPORT_DIR / "completeness_summary.json"

REQUIRED_METADATA_COLUMNS = [
    "drive_id",
    "scene_id",
    "frame_id",
    "timestamp",
    "camera_file",
    "lidar_file",
    "annotation_file",
    "weather",
    "time_of_day",
    "location_type",
]

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path)
    
    missing_columns = [col for col in REQUIRED_METADATA_COLUMNS if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")
    
    return metadata

def check_file_exists(relative_path: str) -> bool:
    if pd.isna(relative_path) or str(relative_path).strip() == "":
        return False
    
    full_path = DATA_DIR / str(relative_path)
    return full_path.exists()

def validate_completeness(metadata: pd.DataFrame) -> pd.DataFrame:
    report = metadata.copy()

    report["has_valid_frame_id"] = report["frame_id"].notna() & (report["frame_id"].astype(str).str.strip() != "")
    report["has_valid_timestamp"] = report["timestamp"].notna() & (report["timestamp"].astype(str).str.strip()!="")

    report["camera_file_exists"] = report["camera_file"].apply(check_file_exists)
    report["lidar_file_exists"] = report["lidar_file"].apply(check_file_exists)
    report["annotation_file_exists"] = report["annotation_file"].apply(check_file_exists)

    duplicated_frame_ids = report["frame_id"].duplicated(keep=False)
    report["is_duplicate_frame_id"] = duplicated_frame_ids

    report["is_complete"] = (
        report["has_valid_frame_id"]
        & report["has_valid_timestamp"]
        & report["camera_file_exists"]
        & report["lidar_file_exists"]
        & report["annotation_file_exists"]
        & (~report["is_duplicate_frame_id"])
    )

    report["failure_reasons"] = report.apply(build_failure_reasons, axis=1)

    return report

def build_failure_reasons(row: pd.Series) -> str:
    reasons = []

    if not row["has_valid_frame_id"]:
        reasons.append("missing_frame_id")

    if not row["has_valid_timestamp"]:
        reasons.append("missing_timestamp")

    if not row["camera_file_exists"]:
        reasons.append("missing_camera_file")

    if not row["lidar_file_exists"]:
        reasons.append("missing_lidar_file")

    if not row["annotation_file_exists"]:
        reasons.append("missing_annotation_file")

    if row["is_duplicate_frame_id"]:
        reasons.append("duplicate_frame_id")

    if not reasons:
        return "passed"

    return ";".join(reasons)

def summarize_completeness(report: pd.DataFrame) -> dict:
    total_frames = len(report)
    complete_frames = int(report["is_complete"].sum())
    incomplete_frames = total_frames - complete_frames

    summary = {
        "total_frames": total_frames,
        "complete_frames": complete_frames,
        "incomplete_frames": incomplete_frames,
        "completion_rate": round(complete_frames / total_frames, 4) if total_frames else 0,
        "missing_timestamp_count": int((~report["has_valid_timestamp"]).sum()),
        "missing_camera_file_count": int((~report["camera_file_exists"]).sum()),
        "missing_lidar_file_count": int((~report["lidar_file_exists"]).sum()),
        "missing_annotation_file_count": int((~report["annotation_file_exists"]).sum()),
        "duplicate_frame_id_count": int(report["is_duplicate_frame_id"].sum()),
    }

    return summary

def run_completeness_validation() -> tuple[pd.DataFrame, dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(METADATA_PATH)
    report = validate_completeness(metadata)
    summary = summarize_completeness(report)

    report.to_csv(COMPLETENESS_REPORT_PATH, index=False)

    with COMPLETENESS_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Completeness validation finished.")
    print(f"Report saved to: {COMPLETENESS_REPORT_PATH}")
    print(f"Summary saved to: {COMPLETENESS_SUMMARY_PATH}")

    return report, summary


if __name__ == "__main__":
    run_completeness_validation()
   

