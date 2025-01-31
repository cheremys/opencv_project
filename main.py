# import cv2
# from hands import HandDetector

# def calculate_water_needed(coffee_count):
#     return (coffee_count * 250) / 1000  # Convert to liters

# def main():
#     cap = cv2.VideoCapture(0)
#     detector = HandDetector()

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
            
#         results = detector.process_frame(frame)
        
#         coffee_count = 0
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 detector.draw_landmarks(frame, hand_landmarks)
#                 coffee_count = detector.count_raised_fingers(hand_landmarks)
        
#         water_needed = calculate_water_needed(coffee_count)
        
#         cv2.putText(frame, f'Coffee cups: {coffee_count}', (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(frame, f'Water needed: {water_needed:.2f}L', (10, 70), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         cv2.imshow('Coffee Counter', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get the output layer names from YOLO
layer_names = net.getUnconnectedOutLayersNames()

# Open webcam
#cap = cv2.VideoCapture(1)  # 0 = default webcam 1 = IPhone webcam
# Open video file
cap = cv2.VideoCapture("video1.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert frame to blob for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through YOLO
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x, y = center_x - w // 2, center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output
    cv2.imshow("YOLO Real-Time Detection", frame)

    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()