from pathlib import Path
import json

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_QUALITY_SUMMARY_PATH = REPORT_DIR / "data_quality_summary.json"
DATA_QUALITY_REPORT_PATH = REPORT_DIR / "data_quality_report.md"

COMPLETENESS_REPORT_PATH = REPORT_DIR / "completeness_report.csv"
IMAGE_QUALITY_REPORT_PATH = REPORT_DIR / "image_quality_report.csv"
ANNOTATION_QUALITY_REPORT_PATH = REPORT_DIR / "annotation_quality_report.csv"
LIDAR_QUALITY_REPORT_PATH = REPORT_DIR / "lidar_quality_report.csv"
SCENARIO_COVERAGE_REPORT_PATH = REPORT_DIR / "scenario_coverage_report.csv"
SCENARIO_COVERAGE_SUMMARY_PATH = REPORT_DIR / "scenario_coverage_summary.json"


st.set_page_config(
    page_title="AV Data Quality Dashboard",
    page_icon="🚗",
    layout="wide",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def render_missing_report_warning() -> None:
    st.warning(
        "Some reports are missing. Please run `python src/run_all_checks.py` "
        "from the project root before launching the dashboard."
    )


def get_release_status_color(status: str) -> str:
    status_lower = status.lower()

    if "ready for ml training release" in status_lower:
        return "green"

    if "conditionally" in status_lower:
        return "orange"

    return "red"


def render_header(summary: dict) -> None:
    st.title("🚗 Autonomous Vehicle Data Quality Dashboard")

    st.markdown(
        """
        This dashboard summarizes automated quality checks for autonomous driving data,
        including sensor completeness, camera image quality, annotation QA, LiDAR point
        cloud validation, and scenario coverage.
        """
    )

    if not summary:
        render_missing_report_warning()
        return

    release_status = summary.get("release_status", "Unknown")
    release_color = get_release_status_color(release_status)

    st.markdown(
        f"""
        ### Release Status: :{release_color}[{release_status}]
        """
    )

    st.caption(f"Generated at: {summary.get('generated_at', 'N/A')}")


def render_kpi_cards(summary: dict) -> None:
    st.subheader("Overall Dataset Readiness")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Frames", summary.get("total_frames", 0))
    col2.metric("Passed All Checks", summary.get("passed_all_checks_count", 0))
    col3.metric("Failed Frames", summary.get("failed_frame_count", 0))
    col4.metric("Overall Pass Rate", summary.get("overall_pass_rate", 0))


def render_pass_rates(summary: dict) -> None:
    st.subheader("Quality Check Pass Rates")

    pass_rate_data = pd.DataFrame(
        [
            {
                "check_category": "Completeness",
                "pass_rate": summary.get("completeness_pass_rate", 0),
            },
            {
                "check_category": "Image Quality",
                "pass_rate": summary.get("image_quality_pass_rate", 0),
            },
            {
                "check_category": "Annotation Quality",
                "pass_rate": summary.get("annotation_quality_pass_rate", 0),
            },
            {
                "check_category": "LiDAR Quality",
                "pass_rate": summary.get("lidar_quality_pass_rate", 0),
            },
        ]
    )

    st.bar_chart(pass_rate_data, x="check_category", y="pass_rate")


def render_failure_counts(summary: dict) -> None:
    st.subheader("Key Failure Counts")

    failure_data = pd.DataFrame(
        [
            {
                "failure_type": "Missing Camera File",
                "count": summary.get("missing_camera_file_count", 0),
            },
            {
                "failure_type": "Missing LiDAR File",
                "count": summary.get("missing_lidar_file_count", 0),
            },
            {
                "failure_type": "Missing Annotation File",
                "count": summary.get("missing_annotation_file_count", 0),
            },
            {
                "failure_type": "Missing Timestamp",
                "count": summary.get("missing_timestamp_count", 0),
            },
            {
                "failure_type": "Image Quality Failure",
                "count": summary.get("image_quality_failed_count", 0),
            },
            {
                "failure_type": "Annotation Quality Failure",
                "count": summary.get("annotation_quality_failed_count", 0),
            },
            {
                "failure_type": "LiDAR Quality Failure",
                "count": summary.get("lidar_quality_failed_count", 0),
            },
        ]
    )

    st.dataframe(failure_data, use_container_width=True)
    st.bar_chart(failure_data, x="failure_type", y="count")


def render_failed_frame_queue(
    completeness_report: pd.DataFrame,
    image_report: pd.DataFrame,
    annotation_report: pd.DataFrame,
    lidar_report: pd.DataFrame,
) -> None:
    st.subheader("Failed Frame Review Queue")

    if (
        completeness_report.empty
        or image_report.empty
        or annotation_report.empty
        or lidar_report.empty
    ):
        st.info("One or more validation reports are missing.")
        return

    failed_rows = []

    def add_failures(df: pd.DataFrame, passed_col: str, reason_col: str, category: str) -> None:
        failed = df[~df[passed_col].astype(bool)]

        for _, row in failed.iterrows():
            failed_rows.append(
                {
                    "frame_id": row["frame_id"],
                    "category": category,
                    "failure_reasons": row.get(reason_col, ""),
                }
            )

    add_failures(completeness_report, "is_complete", "failure_reasons", "completeness")
    add_failures(image_report, "image_quality_passed", "failure_reasons", "image_quality")
    add_failures(annotation_report, "annotation_quality_passed", "failure_reasons", "annotation_quality")
    add_failures(lidar_report, "lidar_quality_passed", "failure_reasons", "lidar_quality")

    failed_frame_df = pd.DataFrame(failed_rows)

    if failed_frame_df.empty:
        st.success("No failed frames detected.")
        return

    selected_category = st.multiselect(
        "Filter by failure category",
        options=sorted(failed_frame_df["category"].unique()),
        default=sorted(failed_frame_df["category"].unique()),
    )

    filtered = failed_frame_df[failed_frame_df["category"].isin(selected_category)]

    st.dataframe(filtered, use_container_width=True)


def render_scenario_coverage(
    scenario_report: pd.DataFrame,
    scenario_summary: dict,
) -> None:
    st.subheader("Scenario Coverage Analysis")

    if scenario_report.empty:
        st.info("Scenario coverage report is not available.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Labeled Objects", scenario_summary.get("total_labeled_objects", 0))
    col2.metric("Avg Objects / Frame", scenario_summary.get("average_objects_per_frame", 0))
    col3.metric("Max Objects / Frame", scenario_summary.get("max_objects_per_frame", 0))
    col4.metric("High-Density Frames", scenario_summary.get("high_object_density_frame_count", 0))

    st.markdown("#### Weather Distribution")
    weather_counts = scenario_report["weather"].value_counts().reset_index()
    weather_counts.columns = ["weather", "count"]
    st.bar_chart(weather_counts, x="weather", y="count")

    st.markdown("#### Time of Day Distribution")
    time_counts = scenario_report["time_of_day"].value_counts().reset_index()
    time_counts.columns = ["time_of_day", "count"]
    st.bar_chart(time_counts, x="time_of_day", y="count")

    st.markdown("#### Location Type Distribution")
    location_counts = scenario_report["location_type"].value_counts().reset_index()
    location_counts.columns = ["location_type", "count"]
    st.bar_chart(location_counts, x="location_type", y="count")

    st.markdown("#### Object Class Distribution")

    class_totals = scenario_summary.get("class_totals", {})

    class_df = pd.DataFrame(
        [
            {"class": class_name, "count": count}
            for class_name, count in class_totals.items()
        ]
    )

    if not class_df.empty:
        st.bar_chart(class_df, x="class", y="count")
    else:
        st.info("No class distribution available.")

    st.markdown("#### Frame-Level Scenario Report")
    st.dataframe(scenario_report, use_container_width=True)


def render_report_tables(
    completeness_report: pd.DataFrame,
    image_report: pd.DataFrame,
    annotation_report: pd.DataFrame,
    lidar_report: pd.DataFrame,
) -> None:
    st.subheader("Detailed Validation Reports")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Completeness",
            "Image Quality",
            "Annotation QA",
            "LiDAR Quality",
        ]
    )

    with tab1:
        if completeness_report.empty:
            st.info("Completeness report is not available.")
        else:
            st.dataframe(completeness_report, use_container_width=True)

    with tab2:
        if image_report.empty:
            st.info("Image quality report is not available.")
        else:
            st.dataframe(image_report, use_container_width=True)

    with tab3:
        if annotation_report.empty:
            st.info("Annotation quality report is not available.")
        else:
            st.dataframe(annotation_report, use_container_width=True)

    with tab4:
        if lidar_report.empty:
            st.info("LiDAR quality report is not available.")
        else:
            st.dataframe(lidar_report, use_container_width=True)


def render_markdown_report() -> None:
    st.subheader("Integrated Markdown Quality Report")

    if not DATA_QUALITY_REPORT_PATH.exists():
        st.info("Integrated markdown report is not available.")
        return

    report_text = DATA_QUALITY_REPORT_PATH.read_text(encoding="utf-8")
    st.markdown(report_text)


def main() -> None:
    summary = load_json(DATA_QUALITY_SUMMARY_PATH)
    scenario_summary = load_json(SCENARIO_COVERAGE_SUMMARY_PATH)

    completeness_report = load_csv(COMPLETENESS_REPORT_PATH)
    image_report = load_csv(IMAGE_QUALITY_REPORT_PATH)
    annotation_report = load_csv(ANNOTATION_QUALITY_REPORT_PATH)
    lidar_report = load_csv(LIDAR_QUALITY_REPORT_PATH)
    scenario_report = load_csv(SCENARIO_COVERAGE_REPORT_PATH)

    render_header(summary)

    if summary:
        render_kpi_cards(summary)
        render_pass_rates(summary)
        render_failure_counts(summary)

    st.divider()

    render_failed_frame_queue(
        completeness_report,
        image_report,
        annotation_report,
        lidar_report,
    )

    st.divider()

    render_scenario_coverage(
        scenario_report,
        scenario_summary,
    )

    st.divider()

    render_report_tables(
        completeness_report,
        image_report,
        annotation_report,
        lidar_report,
    )

    st.divider()

    render_markdown_report()


if __name__ == "__main__":
    main()