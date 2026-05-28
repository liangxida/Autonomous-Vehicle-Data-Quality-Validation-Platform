from pathlib import Path
import json
import cv2  # type: ignore
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "sample"
REPORT_DIR = PROJECT_ROOT / "reports"

METADATA_PATH = DATA_DIR / "metadata.csv"
IMAGE_QUALITY_REPORT_PATH = REPORT_DIR / "image_quality_report.csv"
IMAGE_QUALITY_SUMMARY_PATH = REPORT_DIR / "image_quality_summary.json"


EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 360

MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 230
MIN_CONTRAST = 20
MIN_BLUR_SCORE = 60

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_columns = ["frame_id", "camera_file"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]

    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")

    return metadata

def read_image(image_relative_path: str):
    if pd.isna(image_relative_path) or str(image_relative_path).strip() == "":
        return None

    image_path = DATA_DIR / str(image_relative_path)

    if not image_path.exists():
        return None

    image = cv2.imread(str(image_path))

    return image

def calculate_image_metrics(image: np.ndarray) -> dict:
    """
    Calculate basic image quality metrics.

    brightness:
        Mean grayscale pixel intensity.

    contrast:
        Standard deviation of grayscale pixel intensity.

    blur_score:
        Variance of Laplacian. Lower values generally indicate blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "image_width": int(width),
        "image_height": int(height),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "blur_score": round(blur_score, 4),
    }

def evaluate_image_quality(metrics: dict) -> tuple[bool, str]:
    reasons = []

    if metrics["image_width"] != EXPECTED_WIDTH or metrics["image_height"] != EXPECTED_HEIGHT:
        reasons.append("invalid_resolution")

    if metrics["brightness"] < MIN_BRIGHTNESS:
        reasons.append("underexposed")

    if metrics["brightness"] > MAX_BRIGHTNESS:
        reasons.append("overexposed")

    if metrics["contrast"] < MIN_CONTRAST:
        reasons.append("low_contrast")

    if metrics["blur_score"] < MIN_BLUR_SCORE:
        reasons.append("blurry")

    if not reasons:
        return True, "passed"

    return False, ";".join(reasons)

def validate_images(metadata: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in metadata.iterrows():
        frame_id = row["frame_id"]
        camera_file = row["camera_file"]

        image_path = DATA_DIR / str(camera_file)
        image_exists = image_path.exists()

        result = {
            "frame_id": frame_id,
            "camera_file": camera_file,
            "image_exists": image_exists,
            "image_readable": False,
            "image_width": None,
            "image_height": None,
            "brightness": None,
            "contrast": None,
            "blur_score": None,
            "image_quality_passed": False,
            "failure_reasons": "",
        }

        if not image_exists:
            result["failure_reasons"] = "missing_image_file"
            results.append(result)
            continue

        image = read_image(camera_file)

        if image is None:
            result["failure_reasons"] = "unreadable_or_corrupted_image"
            results.append(result)
            continue

        result["image_readable"] = True

        metrics = calculate_image_metrics(image)
        result.update(metrics)

        passed, failure_reasons = evaluate_image_quality(metrics)
        result["image_quality_passed"] = passed
        result["failure_reasons"] = failure_reasons

        results.append(result)

    return pd.DataFrame(results)

def summarize_image_quality(report: pd.DataFrame) -> dict:
    total_images = len(report)

    image_exists_count = int(report["image_exists"].sum())
    image_readable_count = int(report["image_readable"].sum())
    passed_count = int(report["image_quality_passed"].sum())
    failed_count = total_images - passed_count

    def count_reason(reason: str) -> int:
        return int(report["failure_reasons"].fillna("").str.contains(reason).sum())

    summary = {
        "total_image_records": total_images,
        "image_exists_count": image_exists_count,
        "image_readable_count": image_readable_count,
        "image_quality_passed_count": passed_count,
        "image_quality_failed_count": failed_count,
        "image_quality_pass_rate": round(passed_count / total_images, 4) if total_images else 0,
        "missing_image_file_count": count_reason("missing_image_file"),
        "unreadable_or_corrupted_image_count": count_reason("unreadable_or_corrupted_image"),
        "invalid_resolution_count": count_reason("invalid_resolution"),
        "underexposed_count": count_reason("underexposed"),
        "overexposed_count": count_reason("overexposed"),
        "low_contrast_count": count_reason("low_contrast"),
        "blurry_count": count_reason("blurry"),
    }

    return summary

def run_image_validation() -> tuple[pd.DataFrame, dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(METADATA_PATH)
    report = validate_images(metadata)
    summary = summarize_image_quality(report)

    report.to_csv(IMAGE_QUALITY_REPORT_PATH, index=False)

    with IMAGE_QUALITY_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Image quality validation finished.")
    print(f"Report saved to: {IMAGE_QUALITY_REPORT_PATH}")
    print(f"Summary saved to: {IMAGE_QUALITY_SUMMARY_PATH}")

    return report, summary


if __name__ == "__main__":
    run_image_validation()