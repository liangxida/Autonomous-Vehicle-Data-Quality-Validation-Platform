# Autonomous Vehicle Data Quality Report

Generated at: 2026-05-28T16:55:59

## Executive Summary

This report summarizes automated data quality checks for an autonomous driving dataset, including sensor completeness, camera image quality, annotation quality, and LiDAR point cloud quality.

**Release Status:** Not ready for ML training release

## Dataset Overview

| Metric | Value |
|---|---:|
| Total Frames | 8 |
| Passed All Checks | 2 |
| Failed Frames | 6 |
| Overall Pass Rate | 0.25 |

## Quality Check Pass Rates

| Check Category | Pass Rate |
|---|---:|
| Completeness | 0.5 |
| Image Quality | 0.625 |
| Annotation Quality | 0.75 |
| LiDAR Quality | 0.75 |

## Key Failure Counts

| Failure Type | Count |
|---|---:|
| Missing Camera File | 1 |
| Missing LiDAR File | 1 |
| Missing Annotation File | 1 |
| Missing Timestamp | 1 |
| Image Quality Failures | 3 |
| Annotation Quality Failures | 2 |
| LiDAR Quality Failures | 2 |


## Scenario Coverage Summary

| Metric | Value |
|---|---:|
| Total Labeled Objects | 15 |
| Average Objects per Frame | 1.875 |
| Max Objects per Frame | 3 |
| No-Object Frames | 1 |
| High Object Density Frames | 1 |

## Object Class Distribution

| Class | Count |
|---|---:|
| Vehicle | 13 |
| Pedestrian | 1 |
| Cyclist | 1 |
| Traffic Light | 0 |
| Traffic Sign | 0 |

## Rare Scenario Combinations

No rare scenarios detected.


## Failed Frame Review Queue

| frame_id   | failed_categories                | failure_reasons                                                                                                                                         |
|:-----------|:---------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| frame_003  | annotation_quality;lidar_quality | annotation_quality: invalid_class_label;invalid_bounding_box;frame_id_mismatch | lidar_quality: invalid_numeric_values;nan_or_inf_values;range_outliers |
| frame_004  | completeness;image_quality       | completeness: missing_camera_file | image_quality: missing_image_file                                                                                   |
| frame_005  | completeness;lidar_quality       | completeness: missing_lidar_file | lidar_quality: missing_lidar_file                                                                                    |
| frame_006  | annotation_quality;completeness  | completeness: missing_annotation_file | annotation_quality: missing_annotation_file                                                                     |
| frame_007  | image_quality                    | image_quality: low_contrast                                                                                                                             |
| frame_008  | completeness;image_quality       | completeness: missing_timestamp | image_quality: low_contrast                                                                                           |

## Recommendations

1. Recollect or recover frames with missing sensor files.
2. Correct missing or invalid annotations before model training release.
3. Review low-quality camera frames for blur, exposure, and contrast issues.
4. Investigate LiDAR frames with invalid coordinates, sparse scans, or range outliers.
5. Use this report as a release-readiness gate before downstream ML training.

## Notes

This MVP uses synthetic sample driving-scene data to demonstrate the validation workflow. The same framework can be extended to public or production autonomous driving datasets such as KITTI, BDD100K, nuScenes, or internal test-drive logs.
