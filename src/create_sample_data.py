from pathlib import Path
import csv
import random
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data'/'sample'
IMAGE_DIR = DATA_DIR / 'images'
LIDAR_DIR = DATA_DIR / 'lidar'
ANNOTATION_DIR = DATA_DIR / 'annotations'
METADATA_FILE = DATA_DIR / 'metadata.csv'

APPROVED_CLASSES = [
    'vehicle',
    'pedestrian',
    'cyclist',
    'traffic_light',
    'traffic_sign',
]

def ensure_directories() -> None:
    IMAGE_DIR.mkdir(parents = True, exist_ok = True)
    LIDAR_DIR.mkdir(parents = True, exist_ok = True)
    ANNOTATION_DIR.mkdir(parents = True, exist_ok = True)

def create_sample_image(path: Path, frame_id: str, brightness: int = 120) -> None:
    """
    Create a simple synthetic driving-scene image.

    This is not meant to represent a real autonomous driving dataset.
    It is only used to test image existence, readalibity, and future
    image-quality validation logic.
    """
    width, height = 640, 360

    image = Image.new('RGB', (width, height), color = (brightness, brightness, brightness))
    draw = ImageDraw.Draw(image)

    # Draw road
    draw.polygon(
        [(0, height), (width, height), (410, 210), (230, 210)],
        fill = (70, 70, 70)       
    )

    # Draw lane lines
    draw.line([(320, height), (320, 220)], fill = (255, 255, 255), width = 3)

    # Draw simple vehicles
    draw.rectangle([120, 230, 210, 290], fill = (30, 80, 180))
    draw.rectangle([420, 240, 520, 305], fill = (180, 60, 40))

    # Add frame label
    draw.text((20, 20), frame_id, fill = (255, 255, 255))

    image.save(path)

def create_lidar_file(path: Path, num_points: int = 200) -> None:
    """
    Create a synthetic LiDAR point cloud CSV file.
    
    Columns:
    x, y, z, intensity
    """
    with path.open('w', newline = "", encoding = 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "intensity"])
        
        for _ in range(num_points):
            x = random.uniform(-30, 30)
            y = random.uniform(-5, 80)
            z = random.uniform(-2, 3)
            intensity = random.uniform(0, 1)
            writer.writerow([x, y, z, intensity])

def create_annotation_file(path: Path, frame_id: str, image_width: int = 640, image_height: int = 360) -> None:
    """
    Create a synthetic annotations CSV file.

    Columns:
    frame_id, class, x_min, y_min, x_max, y_max
    """

    objects = [
        [frame_id, 'vehicle', 120, 230, 210, 290],
        [frame_id, 'vehicle', 420, 240, 520, 305]
    ]

    # Randomly add a pedestrain for some frames
    if random.random() > 0.5:
        objects.append([frame_id, 'pedestrian', 300, 220, 330, 310])
    
    with path.open('w', newline = "", encoding = 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "class", "x_min", "y_min", "x_max", "y_max"])
        writer.writerows(objects)
    
def create_metadata() -> None:
    """
    Create metadata for 8 synthetic frames.

    Some intentional quality issues are included:
    - frame_004 has missing image
    - frame_005 has missing LiDAR
    - frame_006 has missing annotations
    - frame_008 has missing timestamp
    """
    rows = [
        {
            "drive_id": "drive_001",
            "scene_id": "scene_001",
            "frame_id": "frame_001",
            "timestamp": "2026-05-01T09:00:00",
            "camera_file": "images/frame_001.jpg",
            "lidar_file": "lidar/frame_001.csv",
            "annotation_file": "annotations/frame_001.csv",
            "weather": "clear",
            "time_of_day": "day",
            "location_type": "urban",
        },
        {
            "drive_id": "drive_001",
            "scene_id": "scene_001",
            "frame_id": "frame_002",
            "timestamp": "2026-05-01T09:00:01",
            "camera_file": "images/frame_002.jpg",
            "lidar_file": "lidar/frame_002.csv",
            "annotation_file": "annotations/frame_002.csv",
            "weather": "clear",
            "time_of_day": "day",
            "location_type": "urban",
        },
        {
            "drive_id": "drive_001",
            "scene_id": "scene_001",
            "frame_id": "frame_003",
            "timestamp": "2026-05-01T09:00:02",
            "camera_file": "images/frame_003.jpg",
            "lidar_file": "lidar/frame_003.csv",
            "annotation_file": "annotations/frame_003.csv",
            "weather": "clear",
            "time_of_day": "day",
            "location_type": "urban",
        },
        {
            "drive_id": "drive_001",
            "scene_id": "scene_002",
            "frame_id": "frame_004",
            "timestamp": "2026-05-01T09:00:03",
            "camera_file": "images/frame_004.jpg",
            "lidar_file": "lidar/frame_004.csv",
            "annotation_file": "annotations/frame_004.csv",
            "weather": "cloudy",
            "time_of_day": "day",
            "location_type": "suburban",
        },
        {
            "drive_id": "drive_001",
            "scene_id": "scene_002",
            "frame_id": "frame_005",
            "timestamp": "2026-05-01T09:00:04",
            "camera_file": "images/frame_005.jpg",
            "lidar_file": "lidar/frame_005.csv",
            "annotation_file": "annotations/frame_005.csv",
            "weather": "cloudy",
            "time_of_day": "day",
            "location_type": "suburban",
        },
        {
            "drive_id": "drive_002",
            "scene_id": "scene_003",
            "frame_id": "frame_006",
            "timestamp": "2026-05-01T20:30:00",
            "camera_file": "images/frame_006.jpg",
            "lidar_file": "lidar/frame_006.csv",
            "annotation_file": "annotations/frame_006.csv",
            "weather": "rain",
            "time_of_day": "night",
            "location_type": "urban",
        },
        {
            "drive_id": "drive_002",
            "scene_id": "scene_003",
            "frame_id": "frame_007",
            "timestamp": "2026-05-01T20:30:01",
            "camera_file": "images/frame_007.jpg",
            "lidar_file": "lidar/frame_007.csv",
            "annotation_file": "annotations/frame_007.csv",
            "weather": "rain",
            "time_of_day": "night",
            "location_type": "urban",
        },
        {
            "drive_id": "drive_002",
            "scene_id": "scene_003",
            "frame_id": "frame_008",
            "timestamp": "",
            "camera_file": "images/frame_008.jpg",
            "lidar_file": "lidar/frame_008.csv",
            "annotation_file": "annotations/frame_008.csv",
            "weather": "rain",
            "time_of_day": "night",
            "location_type": "urban",
        },
    ]

    with METADATA_FILE.open('w', newline = "", encoding = 'utf-8') as f:
        fieldnames = [
            "drive_id",
            "scene_id",
            "frame_id",
            "timestamp",
            "camera_file",
            "lidar_file",
            "annotation_file",
            "weather",
            "time_of_day",
            "location_type",
        ]
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def create_sample_dataset() -> None:
    ensure_directories()
    create_metadata()

    # Create files for all frames first
    for i in range(1, 9):
        frame_id = f"frame_{i:03d}"

        brightness = 120
        if frame_id in {"frame_007", "frame_008"}:
            brightness = 45 # darker night-like images
        
        create_sample_image(IMAGE_DIR / f"{frame_id}.jpg", frame_id, brightness)
        create_lidar_file(LIDAR_DIR / f"{frame_id}.csv", num_points = 200)
        create_annotation_file(ANNOTATION_DIR / f"{frame_id}.csv", frame_id)

    # Intentionally remove some files to create validation failures
    missing_files = [
        IMAGE_DIR / "frame_004.jpg",
        LIDAR_DIR / "frame_005.csv",
        ANNOTATION_DIR / "frame_006.csv",
    ]

    for file_path in missing_files:
        if file_path.exists():
            file_path.unlink()
    
    print("Sample AV dataset created successfully.")
    print(f"Metadata file: {METADATA_FILE}")

if __name__ == "__main__":
    create_sample_dataset()