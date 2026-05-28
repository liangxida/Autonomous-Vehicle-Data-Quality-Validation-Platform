from pathlib import Path
import json
from datetime import datetime
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"

COMPLETENESS_REPORT_PATH = REPORT_DIR / "completeness_report.csv"
IMAGE_QUALITY_REPORT_PATH = REPORT_DIR / "image_quality_report.csv"
ANNOTATION_QUALITY_REPORT_PATH = REPORT_DIR / "annotation_quality_report.csv"
LIDAR_QUALITY_REPORT_PATH = REPORT_DIR / "lidar_quality_report.csv"
SCENARIO_COVERAGE_SUMMARY_PATH = REPORT_DIR / "scenario_coverage_summary.json"

DATA_QUALITY_SUMMARY_PATH = REPORT_DIR / "data_quality_summary.json"
DATA_QUALITY_REPORT_PATH = REPORT_DIR / "data_quality_report.md"


def load_required_report(path: Path, report_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{report_name} not found at {path}. "
            f"Please run the corresponding validation script first."
        )

    return pd.read_csv(path)


def pass_rate(series: pd.Series) -> float:
    total = len(series)
    if total == 0:
        return 0.0
    return round(float(series.sum() / total), 4)


def collect_failed_frames(
    completeness_report: pd.DataFrame,
    image_report: pd.DataFrame,
    annotation_report: pd.DataFrame,
    lidar_report: pd.DataFrame,
) -> pd.DataFrame:
    frames = {}

    def add_failure(frame_id: str, category: str, reason: str) -> None:
        if frame_id not in frames:
            frames[frame_id] = {
                "frame_id": frame_id,
                "failed_categories": [],
                "failure_reasons": [],
            }

        frames[frame_id]["failed_categories"].append(category)
        frames[frame_id]["failure_reasons"].append(f"{category}: {reason}")

    for _, row in completeness_report.iterrows():
        if not bool(row["is_complete"]):
            add_failure(row["frame_id"], "completeness", row["failure_reasons"])

    for _, row in image_report.iterrows():
        if not bool(row["image_quality_passed"]):
            add_failure(row["frame_id"], "image_quality", row["failure_reasons"])

    for _, row in annotation_report.iterrows():
        if not bool(row["annotation_quality_passed"]):
            add_failure(row["frame_id"], "annotation_quality", row["failure_reasons"])

    for _, row in lidar_report.iterrows():
        if not bool(row["lidar_quality_passed"]):
            add_failure(row["frame_id"], "lidar_quality", row["failure_reasons"])

    rows = []

    for frame_id, value in frames.items():
        rows.append(
            {
                "frame_id": frame_id,
                "failed_categories": ";".join(sorted(set(value["failed_categories"]))),
                "failure_reasons": " | ".join(value["failure_reasons"]),
            }
        )

    return pd.DataFrame(rows).sort_values("frame_id") if rows else pd.DataFrame(
        columns=["frame_id", "failed_categories", "failure_reasons"]
    )


def classify_release_status(overall_pass_rate: float) -> str:
    if overall_pass_rate >= 0.95:
        return "Ready for ML training release"
    if overall_pass_rate >= 0.80:
        return "Conditionally ready after targeted review"
    return "Not ready for ML training release"


def build_summary(
    completeness_report: pd.DataFrame,
    image_report: pd.DataFrame,
    annotation_report: pd.DataFrame,
    lidar_report: pd.DataFrame,
    failed_frames: pd.DataFrame,
) -> dict:
    total_frames = len(completeness_report)

    completeness_pass_rate = pass_rate(completeness_report["is_complete"])
    image_pass_rate = pass_rate(image_report["image_quality_passed"])
    annotation_pass_rate = pass_rate(annotation_report["annotation_quality_passed"])
    lidar_pass_rate = pass_rate(lidar_report["lidar_quality_passed"])

    frame_level_pass = (
        completeness_report[["frame_id", "is_complete"]]
        .merge(image_report[["frame_id", "image_quality_passed"]], on="frame_id", how="left")
        .merge(annotation_report[["frame_id", "annotation_quality_passed"]], on="frame_id", how="left")
        .merge(lidar_report[["frame_id", "lidar_quality_passed"]], on="frame_id", how="left")
    )

    frame_level_pass["all_checks_passed"] = (
        frame_level_pass["is_complete"]
        & frame_level_pass["image_quality_passed"]
        & frame_level_pass["annotation_quality_passed"]
        & frame_level_pass["lidar_quality_passed"]
    )

    overall_pass_rate = pass_rate(frame_level_pass["all_checks_passed"])
    release_status = classify_release_status(overall_pass_rate)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_frames": int(total_frames),
        "overall_pass_rate": overall_pass_rate,
        "release_status": release_status,
        "failed_frame_count": int(len(failed_frames)),
        "passed_all_checks_count": int(frame_level_pass["all_checks_passed"].sum()),
        "completeness_pass_rate": completeness_pass_rate,
        "image_quality_pass_rate": image_pass_rate,
        "annotation_quality_pass_rate": annotation_pass_rate,
        "lidar_quality_pass_rate": lidar_pass_rate,
        "missing_camera_file_count": int((~completeness_report["camera_file_exists"]).sum()),
        "missing_lidar_file_count": int((~completeness_report["lidar_file_exists"]).sum()),
        "missing_annotation_file_count": int((~completeness_report["annotation_file_exists"]).sum()),
        "missing_timestamp_count": int((~completeness_report["has_valid_timestamp"]).sum()),
        "image_quality_failed_count": int((~image_report["image_quality_passed"]).sum()),
        "annotation_quality_failed_count": int((~annotation_report["annotation_quality_passed"]).sum()),
        "lidar_quality_failed_count": int((~lidar_report["lidar_quality_passed"]).sum()),
    }

    return summary

        
def render_markdown_report(summary: dict, failed_frames: pd.DataFrame) -> str:
    
    scenario_section = "Scenario coverage summary is not available."

    if SCENARIO_COVERAGE_SUMMARY_PATH.exists():
        with SCENARIO_COVERAGE_SUMMARY_PATH.open("r", encoding="utf-8") as f:
            scenario_summary = json.load(f)

        class_totals = scenario_summary.get("class_totals", {})
        rare_scenarios = scenario_summary.get("rare_scenarios", [])

        rare_scenario_text = "No rare scenarios detected."
        if rare_scenarios:
            rare_scenario_text = "\n".join(
                [
                    f'- {item["scenario_key"]}: {item["frame_count"]} frame(s)'
                    for item in rare_scenarios
                ]
            )

        scenario_section = f"""
## Scenario Coverage Summary

| Metric | Value |
|---|---:|
| Total Labeled Objects | {scenario_summary.get("total_labeled_objects", 0)} |
| Average Objects per Frame | {scenario_summary.get("average_objects_per_frame", 0)} |
| Max Objects per Frame | {scenario_summary.get("max_objects_per_frame", 0)} |
| No-Object Frames | {scenario_summary.get("no_object_frame_count", 0)} |
| High Object Density Frames | {scenario_summary.get("high_object_density_frame_count", 0)} |

## Object Class Distribution

| Class | Count |
|---|---:|
| Vehicle | {class_totals.get("vehicle", 0)} |
| Pedestrian | {class_totals.get("pedestrian", 0)} |
| Cyclist | {class_totals.get("cyclist", 0)} |
| Traffic Light | {class_totals.get("traffic_light", 0)} |
| Traffic Sign | {class_totals.get("traffic_sign", 0)} |

## Rare Scenario Combinations

{rare_scenario_text}
"""
        
    failed_frame_table = "No failed frames detected."

    if not failed_frames.empty:
        failed_frame_table = failed_frames.to_markdown(index=False)

    report = f"""# Autonomous Vehicle Data Quality Report

Generated at: {summary["generated_at"]}

## Executive Summary

This report summarizes automated data quality checks for an autonomous driving dataset, including sensor completeness, camera image quality, annotation quality, and LiDAR point cloud quality.

**Release Status:** {summary["release_status"]}

## Dataset Overview

| Metric | Value |
|---|---:|
| Total Frames | {summary["total_frames"]} |
| Passed All Checks | {summary["passed_all_checks_count"]} |
| Failed Frames | {summary["failed_frame_count"]} |
| Overall Pass Rate | {summary["overall_pass_rate"]} |

## Quality Check Pass Rates

| Check Category | Pass Rate |
|---|---:|
| Completeness | {summary["completeness_pass_rate"]} |
| Image Quality | {summary["image_quality_pass_rate"]} |
| Annotation Quality | {summary["annotation_quality_pass_rate"]} |
| LiDAR Quality | {summary["lidar_quality_pass_rate"]} |

## Key Failure Counts

| Failure Type | Count |
|---|---:|
| Missing Camera File | {summary["missing_camera_file_count"]} |
| Missing LiDAR File | {summary["missing_lidar_file_count"]} |
| Missing Annotation File | {summary["missing_annotation_file_count"]} |
| Missing Timestamp | {summary["missing_timestamp_count"]} |
| Image Quality Failures | {summary["image_quality_failed_count"]} |
| Annotation Quality Failures | {summary["annotation_quality_failed_count"]} |
| LiDAR Quality Failures | {summary["lidar_quality_failed_count"]} |

{scenario_section}

## Failed Frame Review Queue

{failed_frame_table}

## Recommendations

1. Recollect or recover frames with missing sensor files.
2. Correct missing or invalid annotations before model training release.
3. Review low-quality camera frames for blur, exposure, and contrast issues.
4. Investigate LiDAR frames with invalid coordinates, sparse scans, or range outliers.
5. Use this report as a release-readiness gate before downstream ML training.

## Notes

This MVP uses synthetic sample driving-scene data to demonstrate the validation workflow. The same framework can be extended to public or production autonomous driving datasets such as KITTI, BDD100K, nuScenes, or internal test-drive logs.
"""
    return report


def generate_integrated_quality_report() -> tuple[dict, pd.DataFrame]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    completeness_report = load_required_report(COMPLETENESS_REPORT_PATH, "Completeness report")
    image_report = load_required_report(IMAGE_QUALITY_REPORT_PATH, "Image quality report")
    annotation_report = load_required_report(ANNOTATION_QUALITY_REPORT_PATH, "Annotation quality report")
    lidar_report = load_required_report(LIDAR_QUALITY_REPORT_PATH, "LiDAR quality report")

    failed_frames = collect_failed_frames(
        completeness_report,
        image_report,
        annotation_report,
        lidar_report,
    )

    summary = build_summary(
        completeness_report,
        image_report,
        annotation_report,
        lidar_report,
        failed_frames,
    )

    with DATA_QUALITY_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    markdown_report = render_markdown_report(summary, failed_frames)
    DATA_QUALITY_REPORT_PATH.write_text(markdown_report, encoding="utf-8")

    print("Integrated data quality report generated.")
    print(f"Summary saved to: {DATA_QUALITY_SUMMARY_PATH}")
    print(f"Markdown report saved to: {DATA_QUALITY_REPORT_PATH}")

    return summary, failed_frames


if __name__ == "__main__":
    generate_integrated_quality_report()