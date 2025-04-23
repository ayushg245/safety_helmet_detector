# Does the Model Really See the Safety Helmet? A Deep Dive into Explainable Detection

This project combines YOLOv8m-based helmet detection with Explainable AI to not only detect safety helmets in video streams, but also reveal how and why the model makes its predictions.

---

### Key Highlights
- Utilizes the YOLOv8m model for accurate, real-time detection of safety helmets in video frames.
- Enhances model transparency by visualizing which regions of the image influenced detection decisions.
- Processes live or recorded video input to detect helmets frame by frame.
- Designed to support use cases in industrial safety, construction monitoring, and compliance verification.

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

- YOLOv8m-based model trained for helmet detection
- Bounding box colors:
  - 🟦 Blue: safety helmets
- Confidence scores displayed for each detection

---

## 🧠 Explainable AI (XAI) Components

On a normal detection, it may seem like the model is doing exactly what it should — identifying and localizing objects like safety helmets with high confidence. This creates the impression that the model is “seeing” the helmet and making the right decision. But when we look inside the model using Explainable AI techniques like Grad-CAM or EigenCAM, a lot of hidden behavior is revealed.

Explainability helps us visualize what parts of the image the model is actually using to make its predictions. Surprisingly, these attention maps often show the model focusing on:

1. Nearby clothing like vests or faces

2. Shadows or colors that consistently co-occur with helmets in training data

**We dug deep in the last three layers of the model and here are the outputs.**

<h3 align="center">🔍 Visualizing Model Focus with EigenCAM (Last Layer)</h3>

<p align="center">
  <img src="output/Last Layer Output/eigen_cam_last_layer_1.jpg" width="22%" />
  <img src="output/Last Layer Output/eigen_cam_last_layer_2.jpg" width="22%" />
  <img src="output/Last Layer Output/eigen_cam_last_layer_3.jpg" width="22%" />
  <img src="output/Last Layer Output/eigen_cam_last_layer_4.jpg" width="22%" />
</p>


<h3 align="center">🔍 Visualizing Model Focus with EigenCAM (Second Last Layer)</h3>

<p align="center">
  <img src="output/Second Last Layer Output/eigen_cam_secondlast_layer_1.jpg" width="22%" />
  <img src="output/Second Last Layer Output/eigen_cam_secondlast_layer_2.jpg" width="22%" />
  <img src="output/Second Last Layer Output/eigen_cam_secondlast_layer_3.jpg" width="22%" />
  <img src="output/Second Last Layer Output/eigen_cam_secondlast_layer_4.jpg" width="22%" />
</p>

<h3 align="center">🔍 Visualizing Model Focus with EigenCAM (Third Last Layer)</h3>

<p align="center">
  <img src="output/Third Last Layer/eigen_cam_thirdlast_layer_1.jpg" width="22%" />
  <img src="output/Third Last Layer/eigen_cam_thirdlast_layer_2.jpg" width="22%" />
  <img src="output/Third Last Layer/eigen_cam_thirdlast_layer_3.jpg" width="22%" />
  <img src="output/Third Last Layer/eigen_cam_thirdlast_layer_4.jpg" width="22%" />
</p>

One of the most revealing insights from applying Explainable AI techniques like EigenCAM was observing where the model actually pays attention when detecting safety helmets.

## 👀 What We Found

In a large number of predictions, the model doesn't focus solely on the helmet. Instead, the activation maps frequently light up **high-visibility safety vests** worn by the same individuals.

---

### 🧠 Why This Happens

🟡 Helmets and vests often appear together in the training data.  
🟡 The model has learned to associate **vests as a strong context cue** for helmet presence.  
🟡 As a result, it sometimes **relies more on the vest** than the helmet itself to make its decision.

---

### ⚠️ What This Means

While the model still **predicts helmets correctly**, the internal reasoning may not be reliable:

⚠️ It may not truly “understand” what a helmet looks like.  
⚠️ It could **fail to detect helmets** if the vest is absent.  
⚠️ It might even **falsely detect a helmet** when only a vest is present.

---

### 🧠 Why Explainability Matters

This kind of **shortcut learning** — where the model depends on frequently co-occurring but **irrelevant features** — is a well-known behavior in deep learning.  
Using **Explainable AI techniques like Eigen-CAM** helps uncover these hidden dependencies and build **trustworthy, interpretable AI systems**, especially in safety-critical environments.


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
5. After running the python detector, to dive deep into what model sees run the following script, which uses stored images from data folder.
```bash
python eigencam.py
```
6. To experiment, you can add more images in the data folder. 

## Output

The system provides:
1. Real-time video display with detections with bounding boxes and confidence labels
2. Processed video saved to output/ directory
3. For selected frames, Grad-CAM or EigenCAM heatmaps are generated to show where the model is focusing its attention.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [YOLOv8m Hard Hat Detection](https://huggingface.co/keremberke/yolov8m-hard-hat-detection/tree/main)
- OpenCV for computer vision capabilities
- PyTorch for deep learning framework
- XAI research community for explainability concepts
- [Eigen Cam Tutorial](https://jacobgil.github.io/pytorch-gradcam-book/EigenCAM%20for%20YOLO5.html)


