from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "sample"
REPORT_DIR = PROJECT_ROOT / "reports"

METADATA_PATH = DATA_DIR / "metadata.csv"
LIDAR_QUALITY_REPORT_PATH = REPORT_DIR / "lidar_quality_report.csv"
LIDAR_QUALITY_SUMMARY_PATH = REPORT_DIR / "lidar_quality_summary.json"


REQUIRED_LIDAR_COLUMNS = ["x", "y", "z", "intensity"]

MIN_POINT_COUNT = 50
MAX_POINT_COUNT = 10000
MAX_REASONABLE_RANGE = 120.0


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_columns = ["frame_id", "lidar_file"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]

    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")

    return metadata


def read_lidar_file(lidar_relative_path: str) -> pd.DataFrame | None:
    if pd.isna(lidar_relative_path) or str(lidar_relative_path).strip() == "":
        return None

    lidar_path = DATA_DIR / str(lidar_relative_path)

    if not lidar_path.exists():
        return None

    try:
        lidar = pd.read_csv(lidar_path)
        return lidar
    except Exception:
        return None


def validate_single_lidar_file(frame_id: str, lidar_relative_path: str) -> dict:
    lidar_path = DATA_DIR / str(lidar_relative_path)

    result = {
        "frame_id": frame_id,
        "lidar_file": lidar_relative_path,
        "lidar_file_exists": lidar_path.exists(),
        "lidar_file_readable": False,
        "has_required_columns": False,
        "point_count": 0,
        "invalid_numeric_value_count": 0,
        "nan_or_inf_count": 0,
        "mean_range": None,
        "max_range": None,
        "range_outlier_count": 0,
        "lidar_quality_passed": False,
        "failure_reasons": "",
    }

    if not lidar_path.exists():
        result["failure_reasons"] = "missing_lidar_file"
        return result

    lidar = read_lidar_file(lidar_relative_path)

    if lidar is None:
        result["failure_reasons"] = "unreadable_lidar_file"
        return result

    result["lidar_file_readable"] = True

    missing_columns = [
        col for col in REQUIRED_LIDAR_COLUMNS
        if col not in lidar.columns
    ]

    if missing_columns:
        result["failure_reasons"] = "missing_required_lidar_columns"
        return result

    result["has_required_columns"] = True

    if lidar.empty:
        result["failure_reasons"] = "empty_point_cloud"
        return result

    numeric_lidar = lidar.copy()

    invalid_numeric_value_count = 0

    for col in REQUIRED_LIDAR_COLUMNS:
        converted = pd.to_numeric(numeric_lidar[col], errors="coerce")
        invalid_numeric_value_count += int(converted.isna().sum() - numeric_lidar[col].isna().sum())
        numeric_lidar[col] = converted

    result["invalid_numeric_value_count"] = invalid_numeric_value_count

    values = numeric_lidar[REQUIRED_LIDAR_COLUMNS].to_numpy(dtype=float)

    nan_or_inf_mask = ~np.isfinite(values)
    nan_or_inf_count = int(nan_or_inf_mask.sum())
    result["nan_or_inf_count"] = nan_or_inf_count

    # For range calculation, keep only rows with valid x/y/z.
    xyz = numeric_lidar[["x", "y", "z"]].to_numpy(dtype=float)
    valid_xyz_mask = np.isfinite(xyz).all(axis=1)
    valid_xyz = xyz[valid_xyz_mask]

    point_count = len(lidar)
    result["point_count"] = int(point_count)

    if len(valid_xyz) > 0:
        ranges = np.sqrt(
            valid_xyz[:, 0] ** 2
            + valid_xyz[:, 1] ** 2
            + valid_xyz[:, 2] ** 2
        )

        result["mean_range"] = round(float(np.mean(ranges)), 4)
        result["max_range"] = round(float(np.max(ranges)), 4)
        result["range_outlier_count"] = int((ranges > MAX_REASONABLE_RANGE).sum())

    reasons = []

    if point_count < MIN_POINT_COUNT:
        reasons.append("low_point_count")

    if point_count > MAX_POINT_COUNT:
        reasons.append("high_point_count")

    if invalid_numeric_value_count > 0:
        reasons.append("invalid_numeric_values")

    if nan_or_inf_count > 0:
        reasons.append("nan_or_inf_values")

    if result["range_outlier_count"] > 0:
        reasons.append("range_outliers")

    if reasons:
        result["lidar_quality_passed"] = False
        result["failure_reasons"] = ";".join(reasons)
    else:
        result["lidar_quality_passed"] = True
        result["failure_reasons"] = "passed"

    return result


def validate_lidar(metadata: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in metadata.iterrows():
        frame_id = row["frame_id"]
        lidar_file = row["lidar_file"]

        result = validate_single_lidar_file(frame_id, lidar_file)
        results.append(result)

    return pd.DataFrame(results)


def summarize_lidar_quality(report: pd.DataFrame) -> dict:
    total_records = len(report)

    passed_count = int(report["lidar_quality_passed"].sum())
    failed_count = total_records - passed_count

    unreadable_lidar_file_count = int(
        ((~report["lidar_file_readable"]) & report["lidar_file_exists"]).sum()
    )

    summary = {
        "total_lidar_records": total_records,
        "lidar_quality_passed_count": passed_count,
        "lidar_quality_failed_count": failed_count,
        "lidar_quality_pass_rate": round(passed_count / total_records, 4) if total_records else 0,
        "missing_lidar_file_count": int((~report["lidar_file_exists"]).sum()),
        "unreadable_lidar_file_count": unreadable_lidar_file_count,
        "missing_required_lidar_columns_count": int((~report["has_required_columns"] & report["lidar_file_readable"]).sum()),
        "low_point_count": int(report["failure_reasons"].fillna("").str.contains("low_point_count").sum()),
        "high_point_count": int(report["failure_reasons"].fillna("").str.contains("high_point_count").sum()),
        "invalid_numeric_values_count": int(report["invalid_numeric_value_count"].sum()),
        "nan_or_inf_count": int(report["nan_or_inf_count"].sum()),
        "range_outlier_count": int(report["range_outlier_count"].sum()),
    }

    return summary


def run_lidar_validation() -> tuple[pd.DataFrame, dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(METADATA_PATH)
    report = validate_lidar(metadata)
    summary = summarize_lidar_quality(report)

    report.to_csv(LIDAR_QUALITY_REPORT_PATH, index=False)

    with LIDAR_QUALITY_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("LiDAR quality validation finished.")
    print(f"Report saved to: {LIDAR_QUALITY_REPORT_PATH}")
    print(f"Summary saved to: {LIDAR_QUALITY_SUMMARY_PATH}")

    return report, summary


if __name__ == "__main__":
    run_lidar_validation()