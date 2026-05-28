\# Autonomous Vehicle Data Quality Validation Platform



This project builds a Python-based data quality validation workflow for autonomous driving datasets. It validates camera frames, LiDAR point cloud files, annotation files, metadata completeness, image quality, labeling consistency, and dataset release readiness for machine learning model training.



\## Target Role Alignment



This project is designed to demonstrate skills relevant to:



\- Data Quality Specialist

\- Autonomous Vehicle Data Analyst

\- ML Data Validation Analyst

\- Data Annotation QA Analyst

\- Sensor Data QA Analyst

\- ETL / Data Quality Analyst



\## Business Context



Autonomous driving systems depend on high-quality sensor and annotation data. Before collected driving-scene data can be used for model training, it must be checked for completeness, consistency, quality, and labeling correctness.



This project simulates a data quality workflow for autonomous vehicle datasets by validating whether camera, LiDAR, and annotation data are ready for downstream machine learning use.



\## Key Features



\- Driving-scene metadata inventory

\- Camera, LiDAR, and annotation completeness checks

\- Image quality validation

\- LiDAR point cloud quality validation

\- Bounding-box and class-label QA

\- Data quality metric reporting

\- Release-readiness scoring for ML training datasets



\## Project Structure



```text

av-data-quality-validation-platform/

│

├── README.md

├── requirements.txt

│

├── docs/

│   ├── data\\\_quality\\\_standards.md

│   ├── qa\\\_checklist.md

│   └── annotation\\\_guidelines.md

│

├── data/

│   └── sample/

│       ├── metadata.csv

│       ├── images/

│       ├── lidar/

│       └── annotations/

│

├── src/

│   ├── build\\\_metadata.py

│   ├── validate\\\_completeness.py

│   ├── validate\\\_images.py

│   ├── validate\\\_lidar.py

│   ├── validate\\\_annotations.py

│   ├── generate\\\_quality\\\_report.py

│   └── run\\\_all\\\_checks.py

│

├── reports/

│

└── tests/


