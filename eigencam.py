import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Step 1: Load YOLO model
yolo = YOLO("best.pt")           # Your trained YOLOv8 model
model = yolo.model               # Underlying PyTorch model (DetectionModel)

# Step 2: Define a CAM-compatible wrapper up to the safe layers
class CAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = nn.Sequential(*list(model.model.children())[:10])  # up to layer 10 (before Concat)

    def forward(self, x):
        return self.backbone(x)  # only safe layers for CAM

wrapped_model = CAMWrapper(model)

# Step 3: Select a deep convolutional target layer for CAM
target_layers = [list(wrapped_model.backbone.children())[-3]]  # last conv before concat

# Step 4: Load and normalize the image
image_path = "data/image4.jpeg"
img = np.array(Image.open(image_path).resize((640, 640))).astype(np.float32) / 255.0
input_tensor = transforms.ToTensor()(img).unsqueeze(0)  # shape: (1, 3, 640, 640)

# Step 5: Run EigenCAM
with EigenCAM(model=wrapped_model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H, W)
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

# Step 6: Show CAM result
Image.fromarray(cam_image).save(f"output/eigen_cam_thirdlast_layer_4.jpg")
