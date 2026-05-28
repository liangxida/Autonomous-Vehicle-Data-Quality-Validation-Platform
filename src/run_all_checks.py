from validate_completeness import run_completeness_validation
from validate_images import run_image_validation
from validate_annotations import run_annotation_validation


def main() -> None:
    print("Running AV data quality validation checks...")

    run_completeness_validation()
    run_image_validation()
    run_annotation_validation()

    print("All available checks finished.")


if __name__ == "__main__":
    main()