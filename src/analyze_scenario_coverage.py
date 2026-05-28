from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "sample"
REPORT_DIR = PROJECT_ROOT / "reports"

METADATA_PATH = DATA_DIR / "metadata.csv"

SCENARIO_COVERAGE_REPORT_PATH = REPORT_DIR / "scenario_coverage_report.csv"
SCENARIO_COVERAGE_SUMMARY_PATH = REPORT_DIR / "scenario_coverage_summary.json"


APPROVED_CLASSES = {
    "vehicle",
    "pedestrian",
    "cyclist",
    "traffic_light",
    "traffic_sign",
}


HIGH_OBJECT_COUNT_THRESHOLD = 3


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = [
        "frame_id",
        "annotation_file",
        "weather",
        "time_of_day",
        "location_type",
    ]

    missing_columns = [col for col in required_columns if col not in metadata.columns]

    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")

    return metadata


def read_annotation_file(annotation_relative_path: str) -> pd.DataFrame:
    annotation_path = DATA_DIR / str(annotation_relative_path)

    if not annotation_path.exists():
        return pd.DataFrame(
            columns=["frame_id", "class", "x_min", "y_min", "x_max", "y_max"]
        )

    try:
        annotations = pd.read_csv(annotation_path)
    except Exception:
        return pd.DataFrame(
            columns=["frame_id", "class", "x_min", "y_min", "x_max", "y_max"]
        )

    return annotations


def build_frame_level_scenario_report(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in metadata.iterrows():
        frame_id = row["frame_id"]
        annotation_file = row["annotation_file"]

        annotations = read_annotation_file(annotation_file)

        if "class" in annotations.columns and not annotations.empty:
            valid_class_annotations = annotations[
                annotations["class"].isin(APPROVED_CLASSES)
            ]

            object_count = len(valid_class_annotations)

            class_counts = (
                valid_class_annotations["class"]
                .value_counts()
                .to_dict()
            )
        else:
            object_count = 0
            class_counts = {}

        output_row = {
            "frame_id": frame_id,
            "weather": row["weather"],
            "time_of_day": row["time_of_day"],
            "location_type": row["location_type"],
            "object_count": object_count,
            "has_objects": object_count > 0,
            "is_high_object_density": object_count >= HIGH_OBJECT_COUNT_THRESHOLD,
            "vehicle_count": class_counts.get("vehicle", 0),
            "pedestrian_count": class_counts.get("pedestrian", 0),
            "cyclist_count": class_counts.get("cyclist", 0),
            "traffic_light_count": class_counts.get("traffic_light", 0),
            "traffic_sign_count": class_counts.get("traffic_sign", 0),
        }

        output_row["scenario_key"] = (
            f'{output_row["weather"]}_'
            f'{output_row["time_of_day"]}_'
            f'{output_row["location_type"]}'
        )

        rows.append(output_row)

    return pd.DataFrame(rows)


def summarize_distribution(df: pd.DataFrame, column: str) -> dict:
    counts = df[column].fillna("missing").value_counts().to_dict()

    total = len(df)
    percentages = {
        key: round(value / total, 4) if total else 0
        for key, value in counts.items()
    }

    return {
        "counts": counts,
        "percentages": percentages,
    }


def identify_rare_scenarios(frame_report: pd.DataFrame) -> list[dict]:
    scenario_counts = (
        frame_report["scenario_key"]
        .value_counts()
        .reset_index()
    )

    scenario_counts.columns = ["scenario_key", "frame_count"]

    rare_scenarios = scenario_counts[
        scenario_counts["frame_count"] == 1
    ]

    return rare_scenarios.to_dict(orient="records")


def build_scenario_summary(frame_report: pd.DataFrame) -> dict:
    total_frames = len(frame_report)

    class_totals = {
        "vehicle": int(frame_report["vehicle_count"].sum()),
        "pedestrian": int(frame_report["pedestrian_count"].sum()),
        "cyclist": int(frame_report["cyclist_count"].sum()),
        "traffic_light": int(frame_report["traffic_light_count"].sum()),
        "traffic_sign": int(frame_report["traffic_sign_count"].sum()),
    }

    no_object_frame_count = int((~frame_report["has_objects"]).sum())
    high_density_frame_count = int(frame_report["is_high_object_density"].sum())

    summary = {
        "total_frames": int(total_frames),
        "weather_distribution": summarize_distribution(frame_report, "weather"),
        "time_of_day_distribution": summarize_distribution(frame_report, "time_of_day"),
        "location_type_distribution": summarize_distribution(frame_report, "location_type"),
        "scenario_distribution": summarize_distribution(frame_report, "scenario_key"),
        "rare_scenarios": identify_rare_scenarios(frame_report),
        "total_labeled_objects": int(frame_report["object_count"].sum()),
        "average_objects_per_frame": round(float(frame_report["object_count"].mean()), 4) if total_frames else 0,
        "max_objects_per_frame": int(frame_report["object_count"].max()) if total_frames else 0,
        "no_object_frame_count": no_object_frame_count,
        "high_object_density_frame_count": high_density_frame_count,
        "class_totals": class_totals,
    }

    return summary


def run_scenario_coverage_analysis() -> tuple[pd.DataFrame, dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    frame_report = build_frame_level_scenario_report(metadata)
    summary = build_scenario_summary(frame_report)

    frame_report.to_csv(SCENARIO_COVERAGE_REPORT_PATH, index=False)

    with SCENARIO_COVERAGE_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Scenario coverage analysis finished.")
    print(f"Report saved to: {SCENARIO_COVERAGE_REPORT_PATH}")
    print(f"Summary saved to: {SCENARIO_COVERAGE_SUMMARY_PATH}")

    return frame_report, summary


if __name__ == "__main__":
    run_scenario_coverage_analysis()