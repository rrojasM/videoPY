import sys
from imutils.video import VideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    parser.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    parser.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
    return vars(parser.parse_args())

def load_model(prototxt, model):
    print("Loading Neural Network...")
    return cv2.dnn.readNetFromCaffe(prototxt, model)

def detect_objects(net, frame, confidence_threshold, classes, colors):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{classes[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    return frame

def main():
    args = parse_arguments()
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "monitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    net = load_model(args["prototxt"], args["model"])

    print("Starting Camera...")
    vs = VideoStream(src=0, resolution=(1600, 1200)).start()
    time.sleep(2.0)
    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        frame = detect_objects(net, frame, args["confidence"], CLASSES, COLORS)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        fps.update()

    fps.stop()
    print(f"[INFO] elapsed time: {fps.elapsed():.2f}")
    print(f"[INFO] approx. FPS: {fps.fps():.2f}")

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()