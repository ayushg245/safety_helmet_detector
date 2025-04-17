# Safety Helmet Detection with Explainable AI

A computer vision system that detects safety helmets in video streams using YOLOv3, enhanced with Explainable AI (XAI) components for better understanding and trust in the detection process.

[Based on the YOLOv3 Helmet Detection implementation by BlcaKHat](https://github.com/BlcaKHat/yolov3-Helmet-Detection)

## Project Overview

This project implements a real-time safety helmet detection system with the following key components:

1. **Computer Vision System**
   - Real-time safety helmet detection in video streams
   - Processes every 10th frame for efficient performance
   - Visual indicators for safety helmet detection
   - Confidence scores for each detection

2. **Explainable AI Components**
   - Model interpretability through confidence scores
   - Visual explanations of detection decisions
   - Transparency in detection process
   - Trust-building through clear feedback

## Video Examples

### Input Video
The system processes video input from the `data` directory:

<video width="640" height="360" controls>
  <source src="data/video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Output Video
The processed video with safety helmet detections is saved in the `output` directory:

<video width="640" height="360" controls>
  <source src="output/safety_helmet_detection.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Features

### Detection Features
- Real-time safety helmet detection in video streams
- Processes every 10th frame for efficient performance
- Visual indicators for safety helmet detection:
  - Green bounding boxes for detected safety helmets
  - Blue bounding boxes for other detected objects
  - Confidence scores displayed for each detection
- Only displays frames where detections are found
- Simple user interface with 'q' to quit

### XAI Features
- Confidence score visualization
- Clear labeling of detection results
- Visual feedback for detection decisions
- Transparent processing pipeline
- Trust indicators through consistent detection patterns

## Technical Implementation

### Model Architecture
- YOLOv3-based detection system
- Custom-trained model for safety helmet detection
- Confidence threshold of 0.35 for reliable detections
- Frame skipping for efficient processing

### XAI Implementation
- Confidence score display for each detection
- Visual bounding boxes with clear labels
- Processing information feedback
- Detection decision transparency

## Requirements

- Python 3.8 or higher
- PyTorch
- OpenCV
- Ultralytics (for YOLOv3)
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

Or install the packages individually:
```bash
pip install torch ultralytics opencv-python numpy
```

## Usage

1. Place your video file in the data folder (you can use video.mp4)
2. Run the script:
```bash
python helmet_detector.py
```
3. Enter the path to your video file when prompted
4. Press 'q' to quit the detection window

## Output

The system provides:
1. Real-time video display with detections
2. Saved output video in the "output" directory
3. Visual feedback for each detection
4. Confidence scores for decision transparency

## XAI Components

### 1. Model Interpretability
- Confidence scores for each detection
- Clear labeling of detection results
- Visual feedback for detection decisions

### 2. Transparency
- Processing information display
- Detection decision explanations
- Clear visual indicators

### 3. Trust Building
- Consistent detection patterns
- Reliable confidence scoring
- Clear visual feedback

## Performance

- Processes every 10th frame to maintain real-time performance
- Uses confidence threshold of 0.35 for reliable detections
- Can run on both CPU and GPU (with CUDA support)
- Efficient processing with frame skipping

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [YOLOv3 Helmet Detection by BlcaKHat](https://github.com/BlcaKHat/yolov3-Helmet-Detection)
- YOLOv3 by Joseph Redmon and Ali Farhadi
- OpenCV for computer vision capabilities
- PyTorch for deep learning framework
- XAI research community for explainability concepts