from validate_completeness import run_completeness_validation
from validate_images import run_image_validation
from validate_annotations import run_annotation_validation
from validate_lidar import run_lidar_validation
from generate_quality_report import generate_integrated_quality_report


def main() -> None:
    print("Running AV data quality validation checks...")

    run_completeness_validation()
    run_image_validation()
    run_annotation_validation()
    run_lidar_validation()
    generate_integrated_quality_report()

    print("All checks and integrated report generation finished.")


if __name__ == "__main__":
    main()