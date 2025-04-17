import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect, Bottleneck, C3
from torch.nn import Module, ModuleList, Upsample, MaxPool2d
from ultralytics.utils.torch_utils import fuse_conv_and_bn


def load_model():
    try:
        # Add all required classes to safe globals
        torch.serialization.add_safe_globals(
            [
                DetectionModel,
                Sequential,
                Conv,
                C2f,
                SPPF,
                Detect,
                Module,
                ModuleList,
                Bottleneck,
                C3,
                Upsample,
                MaxPool2d,
                fuse_conv_and_bn,
            ]
        )

        # Check if model file exists
        if not os.path.exists("best.pt"):
            print("Error: best.pt model file not found!")
            return None

        # Load the model
        model = YOLO("best.pt")
        print("Model loaded successfully!")
        print("Available classes:", model.names)  # Print available classes
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def process_video(video_path):
    # Load helmet detection model
    model = load_model()
    if model is None:
        return

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Create VideoWriter object
    output_path = output_dir / "safety_helmet_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID' for .avi
    out = cv2.VideoWriter(str(output_path), fourcc, fps / 10, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 10th frame
        if frame_count % 10 != 0:
            continue

        # Run detection
        results = model(frame, conf=0.35)[0]

        # Check for detections
        detections_found = False

        # Process detections
        for result in results.boxes.data:
            conf = float(result[4])
            cls = int(result[5])
            box = result[:4].cpu().numpy().astype(int)

            # Get class name
            class_name = model.names[cls]

            # Set color and label based on class
            if "helmet" in class_name.lower() or "hard hat" in class_name.lower():
                color = (0, 255, 0)  # Green for safety helmet
                label = "Safety Helmet"
            else:
                color = (255, 0, 0)  # Blue for other detections
                label = class_name

            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Add label and confidence
            cv2.putText(
                frame,
                f"{label}: {conf:.2f}",
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            detections_found = True

        # Only display and save frame if detections are found
        if detections_found:
            cv2.imshow("Safety Helmet Detection", frame)
            out.write(frame)  # Save the frame to video

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to: {output_path}")


if __name__ == "__main__":
    video_path = input("Enter the path to your video file: ")
    process_video(video_path)
