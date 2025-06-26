# iSAID Preprocessing and YOLO Conversion Toolkit

A complete, step-by-step toolkit to preprocess the iSAID (Instance-Level Semantic Annotation for Aerial Images) dataset. The scripts guide you through splitting the large aerial images into smaller patches, generating annotations in the standard COCO format, and optionally converting the entire dataset into the segmentation format required by Ultralytics YOLO.

**Author:** Mridankan Mandal

## Features

-   **Image Patching**: Splits the large source images into smaller, overlapping patches suitable for training.
-   **COCO Annotation Generation**: Creates COCO-style JSON annotation files from the iSAID instance masks for the training and validation sets.
-   **Test Set Handling**: Generates a COCO-compliant JSON file for the test set images (without labels).
-   **YOLO Format Conversion**: Provides a script to convert the COCO-formatted dataset into the YOLO segmentation format, including `.txt` label files and the required `data.yaml`.

## Prerequisites

1.  **Python 3.6+ (Tested on Python 3.6 and Python 3.11)** 
2.  The iSAID dataset. Download it from the [official website](https://captain-whu.github.io/iSAID/).
3.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
**Note**: This Toolkit has been tested extensively on Windows 11, and Python 3.11.

## Directory Setup

Before you begin, you must organize your downloaded iSAID dataset into the following structure:

```
iSAID_dataset/
├── train/
│   └── images/
│       ├── P0002.png
│       ├── P0002_instance_color_RGB.png
│       ├── P0002_instance_id_RGB.png
│       └── ...
├── val/
│   └── images/
│       ├── P0001.png
│       ├── P0001_instance_color_RGB.png
│       ├── P0001_instance_id_RGB.png
│       └── ...
└── test/
    └── images/
        ├── P0003.png
        └── ...
```

For the default commands to work, download and place the `iSAID_dataset` folder inside this project's root directory, as shown below:

```
new_iSAID_Toolkit/
├── iSAID_dataset/            <- Place raw dataset here
│   ├── train/
│   │   └── images/
│   ├── val/
│   │   └── images/
│   └── test/
│       └── images/
├── split.py
├── preprocess.py
├── generate_test_json.py
├── convert_to_yolo.py
├── requirements.txt
└── README.md
```

## Usage Workflow

Follow these steps in order to process the dataset. Each step includes a simple command that relies on the default directory structure, and a second, more explicit command that shows all parameters.

### Step 1: Split Large Images into Patches

This script creates smaller, overlapping patches and places them in a new `iSAID_patches` directory.

* **Command (using defaults):**
    ```bash
    python split.py
    ```

* **Command (with explicit arguments):**
    ```bash
    python split.py --src ./iSAID_dataset --tar ./iSAID_patches --patch_width 800 --patch_height 800 --overlap_area 200 --set train,val,test
    ```

### Step 2: Generate COCO Annotations

This step creates COCO-style JSON annotation files for the `train` and `val` sets.

* **Command (using defaults):**
    ```bash
    python preprocess.py
    ```

* **Command (with explicit arguments):**
    ```bash
    python preprocess.py --datadir ./iSAID_patches --outdir ./iSAID_patches --set train,val
    ```

### Step 3: Generate Test Set JSON File

This creates a JSON file for the test images, which is useful for a consistent dataset structure.

* **Command (using defaults):**
    ```bash
    python generate_test_json.py
    ```

* **Command (with explicit arguments):**
    ```bash
    python generate_test_json.py --datadir ./iSAID_patches --outdir ./iSAID_patches --set test
    ```

### Step 4 (Optional): Convert to YOLO Segmentation Format

If you intend to train a YOLO segmentation model, this final script converts the COCO-formatted data into the required YOLO format.

* **Command (using defaults):**
    ```bash
    python convert_to_yolo.py
    ```

* **Command (with explicit arguments):**
    ```bash
    python convert_to_yolo.py --datadir ./iSAID_patches --outdir ./iSAID_YOLO_Dataset
    ```

## Final Output Structure

After running all the steps, you will have two primary output directories:

1.  **`./iSAID_patches`**: The dataset in COCO format, ready for use with frameworks like Detectron2, MMDetection, etc.
    ```
    iSAID_patches/
    ├── train/
    │   ├── images/
    │   └── instancesonly_filtered_train.json
    ├── val/
    │   ├── images/
    │   └── instancesonly_filtered_val.json
    └── test/
        ├── images/
        └── instancesonly_filtered_test.json
    ```
2.  **`./iSAID_YOLO_Dataset`**: The dataset in YOLOv8 segmentation format, ready for training with Ultralytics.
    ```
    iSAID_YOLO_Dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
    ```

## Acknowledgments

-   This toolkit was created by Mridankan Mandal.
-   This toolkit is designed for the [iSAID dataset](https://captain-whu.github.io/iSAID/). Please cite the original authors if you use this dataset in your research.