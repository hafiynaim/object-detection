import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained model
MODEL_PATH = 'ssd_mobilenet_v2_fpnlite_320x320/saved_model'
print("Loading model...")
detect_fn = tf.saved_model.load(MODEL_PATH)
print("Model loaded successfully!")

# Load label map
LABEL_MAP = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light'
}

# Load video or webcam
video = cv2.VideoCapture(0)  # Use 'video.mp4' for file input, 0 for webcam

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert image to tensor
    input_tensor = tf.convert_to_tensor([frame])
    input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.uint8)

    # Perform detection
    detections = detect_fn(input_tensor)

    # Extract detection data
    bboxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    h, w, _ = frame.shape

    # Loop through detections
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            # Bounding box
            ymin, xmin, ymax, xmax = bboxes[i]
            x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

            # Label
            label = LABEL_MAP.get(classes[i], 'Unknown')

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ({int(scores[i] * 100)}%)', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Object Detection', frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
