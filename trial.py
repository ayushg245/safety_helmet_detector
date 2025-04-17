import cv2 as cv
import numpy as np
import os
from glob import glob

# -------------------- Settings --------------------
confThreshold = 0.3
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416
output_folder = "test_out"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# -------------------- Load class labels --------------------
classesFile = "obj.names"
with open(classesFile, "rt") as f:
    classes = f.read().rstrip("\n").split("\n")

# -------------------- Load YOLOv3 model --------------------
modelConfiguration = "yolov3-obj.cfg"
modelWeights = "yolov3-obj_2400.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# -------------------- Global frame counters --------------------
frame_count = 0
frame_count_out = 0


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


def drawPred(classId, conf, left, top, right, bottom):
    global frame, frame_count

    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = "%.2f" % conf
    label_text = f"{classes[classId]}:{label}"

    labelSize, baseLine = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    if classes[classId].lower() == "helmet":  # Case-insensitive match
        cv.rectangle(
            frame,
            (left, top - round(1.5 * labelSize[1])),
            (left + round(1.5 * labelSize[0]), top + baseLine),
            (255, 255, 255),
            cv.FILLED,
        )
        cv.putText(
            frame, label_text, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1
        )
        frame_count += 1

    return frame_count


def postprocess(frame, outs, filename):
    global frame_count_out
    frameHeight, frameWidth = frame.shape[:2]
    frame_count_out = 0
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_helmet = 0

    for i in indices.flatten():
        box = boxes[i]
        left, top, width, height = box
        frame_count_out = drawPred(
            classIds[i], confidences[i], left, top, left + width, top + height
        )

        detected_class = classes[classIds[i]]
        print(f"Detected: {detected_class}, Confidence: {confidences[i]:.2f}")

        if detected_class.lower() == "helmet":
            count_helmet += 1

    if count_helmet >= 1:
        print(f"✅ Helmet Detected — Saving {filename}")
        cv.imwrite(os.path.join(output_folder, filename), frame)

    # Show image for debug (comment out if running many images)
    cv.imshow("Helmet Detection", frame)
    cv.waitKey(500)


# -------------------- Run on images --------------------
cv.namedWindow("Helmet Detection", cv.WINDOW_NORMAL)

for fn in glob("images/*.jpg"):
    frame = cv.imread(fn)
    frame_name = os.path.basename(fn)
    frame_count = 0

    blob = cv.dnn.blobFromImage(
        frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False
    )
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs, frame_name)

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
