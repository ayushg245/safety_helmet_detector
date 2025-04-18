# Safety Helmet Detection with Explainable AI

A computer vision system that detects safety helmets in video streams using YOLOv3, enhanced with Explainable AI (XAI) components to improve trust and transparency in model decisions.

> Based on the [YOLOv3 Helmet Detection implementation by BlcaKHat](https://github.com/BlcaKHat/yolov3-Helmet-Detection)

---

## 📌 Project Overview

This system performs real-time safety helmet detection in videos using a YOLOv3-based deep learning model. It incorporates Explainable AI (XAI) components to offer interpretability through visual feedback and confidence scores.

### Key Highlights
- Detects helmets and humans with bounding boxes and confidence levels
- Efficient processing: analyzes every 10th frame
- Real-time visualization with minimal performance drop
- Explainable outputs to build user trust

## 🎥 Input & Output
 
Place your video in the `data` directory:

Processed video is saved in the `output` directory:

## Result

### Input Video
The system processes video input from the `data` directory:

![Input Video](data/video.mp4)

### Helmet Detection Output Video
The processed video with safety helmet detections is saved in the `output` directory:

![Output Video](output/safety_helmet_detection.mp4)

## ⚙️ Features

### Detection
- YOLOv3-based model trained for helmet detection
- Bounding box colors:
  - 🟩 Green: safety helmets
  - 🟦 Blue: other detected objects
- Confidence scores displayed for each detection
- Only displays frames where humans are detected

### Performance
- Processes every 10th frame to balance speed and accuracy
- Runs on both CPU and GPU (CUDA-supported)
- Uses a confidence threshold of 0.35 for reliable detection

---

## 🧠 Explainable AI (XAI) Components

### Model Interpretability
- Visual bounding boxes
- Confidence scores on each detection
- Frame-level feedback for decision reasoning

### Transparency & Trust
- Clear labeling of detection results
- Consistent detection patterns
- Easy-to-understand visual output

## 🛠️ Technical Overview

### Architecture
- YOLOv3-based object detection
- Custom-trained for helmet detection
- Lightweight, efficient Python script


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

1. Place your video file in the data folder (you can use data/video.mp4 for a trial for this project)
2. Run the script:
```bash
python helmet_detector.py
```
3. Enter the path to your video file when prompted (data/video.mp4)
4. Press 'q' to quit the detection window

## Output

The system provides:
1. Real-time video display with detections
2. Processed video saved to output/ directory
3. Bounding boxes and confidence labels shown for detected objects


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