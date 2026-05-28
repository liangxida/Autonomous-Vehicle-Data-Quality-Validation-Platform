\# Data Quality Standards



\## 1. Sensor Completeness



A driving-scene frame is considered complete if it has:



\- A valid frame ID

\- A valid timestamp

\- A readable camera image

\- A non-empty LiDAR point cloud file

\- A corresponding annotation file



\## 2. Camera Data Quality



Camera images are checked for:



\- File readability

\- Resolution

\- Brightness

\- Contrast

\- Blur

\- Corruption



Potential quality issues include:



\- Missing image file

\- Corrupted image file

\- Severe blur

\- Underexposure

\- Overexposure

\- Low contrast

\- Unexpected resolution



\## 3. LiDAR Data Quality



LiDAR point cloud files are checked for:



\- Non-empty point count

\- Valid numeric coordinates

\- No NaN or infinite values

\- Reasonable point count range

\- Reasonable distance/range distribution



Potential quality issues include:



\- Missing LiDAR file

\- Empty point cloud

\- Invalid coordinates

\- Abnormally low point count

\- Abnormally high point count



\## 4. Annotation Quality



Annotation files are checked for:



\- Valid object class labels

\- Bounding boxes inside image boundaries

\- Positive bounding box width and height

\- Object count consistency

\- Explicit handling of frames with no objects



Approved classes for this MVP:



\- vehicle

\- pedestrian

\- cyclist

\- traffic\_light

\- traffic\_sign



\## 5. Release Readiness



A dataset is considered ready for downstream ML training if:



\- Critical missing files are below the defined threshold

\- Corrupted sensor files are removed or repaired

\- Invalid annotations are corrected

\- Data quality report is generated

\- Known limitations are documented

