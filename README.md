# Smart Office System - Dataset Labeling Tools

This branch contains tools for downloading and labeling images for the Smart Office Security System. These tools help create a training and testing dataset for the person detection model.

## Contents

1. **download_dataset.py** - Script to download sample images of people from Unsplash
2. **GroundingDINO.py** - Implementation of the Grounding DINO model for object detection
3. **run_labeling.py** - Script to run the Grounding DINO model on a dataset of images

## Requirements

These scripts require Python and the following dependencies:

```bash
# Create and activate a conda environment (recommended)
conda create -n smart_office python=3.8
conda activate smart_office

# Install required packages
pip install torch torchvision pillow requests transformers
```

## Usage

### 1. Download Sample Images

To download a set of sample images containing people:

```bash
python download_dataset.py
```

This will download 10 images from Unsplash and save them to the `dataset/images` directory.

### 2. Label Images

To label the downloaded images with person detections:

```bash
python run_labeling.py
```

This script will:
- Process all images in the `dataset/images` directory
- Detect people in the images using the Grounding DINO model
- Save annotated images to `dataset/output/annotated_images`
- Save XML annotations in PASCAL VOC format to `dataset/output/xml_annotations`

## Labeling Configuration

The labeling tool is configured to detect people with the following parameters:

- **Text Queries**: "person . human . man . woman . people . individual"
- **Confidence Threshold**: 0.10
- **Box Threshold**: 0.25
- **Text Threshold**: 0.25

You can modify these parameters in the `GroundingDINO.py` and `run_labeling.py` files.

## Output Format

### Annotated Images

The tool saves annotated images with bounding boxes drawn around detected people. Each box is labeled with the detected class and confidence score.

### XML Annotations

The tool generates XML annotations in PASCAL VOC format, which can be used for training object detection models. Each XML file contains:

- Image information (filename, size)
- Object annotations (class, bounding box coordinates)

## How It Works

1. The tool loads the Grounding DINO model from HuggingFace
2. It processes each image with the model, detecting objects based on text queries
3. The detected objects are filtered by confidence score
4. Bounding boxes are drawn on the images and saved as annotated images
5. The annotations are converted to XML format and saved

## Troubleshooting

If you encounter issues with detections:

1. Try adjusting the confidence threshold in `GroundingDINO.py`
2. Make sure the images contain clearly visible people
3. Try different text queries in `run_labeling.py`

## Additional Information

The Grounding DINO model is a zero-shot object detector that can detect objects based on text descriptions. It can be used to detect various objects beyond people, such as "hard hat", "high-visibility vest", etc., by modifying the text queries. 