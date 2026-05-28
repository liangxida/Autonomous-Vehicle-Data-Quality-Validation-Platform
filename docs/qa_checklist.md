\# AV Data Quality QA Checklist



\## Completeness



\- \[ ] All frame IDs are unique

\- \[ ] All timestamps are valid

\- \[ ] All image files exist

\- \[ ] All image files are readable

\- \[ ] All LiDAR files exist

\- \[ ] All LiDAR files are non-empty

\- \[ ] All annotation files exist



\## Camera Quality



\- \[ ] No corrupted images

\- \[ ] No severely blurry images

\- \[ ] No severely underexposed images

\- \[ ] No severely overexposed images

\- \[ ] No invalid image resolutions



\## LiDAR Quality



\- \[ ] No empty point clouds

\- \[ ] No invalid coordinates

\- \[ ] No NaN or infinite values

\- \[ ] Point count distribution is reasonable



\## Annotation QA



\- \[ ] Bounding boxes are inside image boundaries

\- \[ ] Bounding boxes have positive width and height

\- \[ ] Class labels belong to the approved class list

\- \[ ] Frames with no objects are handled consistently

\- \[ ] Object count distribution is reviewed



\## Release Readiness



\- \[ ] Critical errors are resolved

\- \[ ] Rejected frames are documented

\- \[ ] Quality report is generated

\- \[ ] Dataset limitations are documented

\- \[ ] Dataset is ready for downstream ML training or manual review

