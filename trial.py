import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import os


# Grad-CAM Class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.model
        self.target_layer = dict(self.model.named_modules())[target_layer]
        self.gradients = None
        self.activations = None
        self._register_hooks()
        self.target_layer_name = target_layer  # for debug

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        original_mode = self.model.training

        try:
            self.model.train()
            self.model.zero_grad()
            input_tensor.requires_grad_()

            with torch.set_grad_enabled(True):
                raw_output = self.model.forward(input_tensor)[0]
                score = raw_output[..., 4].max()

                if not score.requires_grad:
                    raise RuntimeError(f"Score does not require grad.")

                score.backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                raise RuntimeError(f"Grad-CAM failed. Layer '{self.target_layer_name}' did not activate.")

            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.nn.functional.relu(cam)
            cam = cam.squeeze().detach().cpu().numpy()
            cam_min, cam_max = cam.min(), cam.max()
            print(f"[Grad-CAM] min: {cam_min:.4f}, max: {cam_max:.4f}")
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            return cam

        finally:
            if not original_mode:
                self.model.eval()


# Load the YOLO model
def load_model():
    if not os.path.exists("best.pt"):
        print("Error: best.pt model file not found!")
        return None
    try:
        model = YOLO("best.pt")
        print("Model loaded successfully!")
        print("Available classes:", model.names)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


# Process video and apply detection + Grad-CAM
def process_video(video_path):
    det_model = load_model()
    gradcam_model = load_model()
    if det_model is None or gradcam_model is None:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "helmet_detection_with_gradcam.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))

    frame_count = 0
    grad_cam = GradCAM(gradcam_model, target_layer="model.9.cv1.conv")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue

        results = det_model(frame, conf=0.35)[0]
        detections_found = False

        for result in results.boxes.data:
            conf = float(result[4])
            cls = int(result[5])
            box = result[:4].cpu().numpy().astype(int)
            class_name = det_model.names[cls]

            if "helmet" in class_name.lower() or "hard hat" in class_name.lower():
                color = (0, 255, 0)
                label = "Safety Helmet"
            else:
                color = (255, 0, 0)
                label = class_name

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
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

        if detections_found:
            preprocess = Compose([
                Resize((640, 640)),
                ToTensor()
            ])
            input_tensor = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).float().to(gradcam_model.device)

            try:
                cam = grad_cam.generate(input_tensor)

                cam_resized = cv2.resize(cam, (width, height))

                # Stretch low values to mid/high range
                amplified_cam  = np.clip(cam_resized * 1000, 0, 1)  # Multiply for visibility
                # cv2.putText(frame, f"Grad-CAM max: {cam.max():.4f}", (10, 30),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Convert to heatmap
                heatmap = np.uint8(255 * amplified_cam )
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Blend heatmap with original image
                overlay = cv2.addWeighted(frame, 0.3, heatmap, 0.7, 0)
                combined = np.hstack((frame, overlay))
                cv2.imshow("YOLO Detection (Left) vs Grad-CAM (Right)", combined)
                for _ in range(10):
                    out.write(combined)


            except Exception as e:
                print(f"[Grad-CAM Error] {e}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {output_path}")


# Entry point
if __name__ == "__main__":
    video_path = 'data/video.mp4'  # Replace with your video path
    process_video(video_path)
