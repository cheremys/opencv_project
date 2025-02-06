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

# import cv2
# import numpy as np

# # Load YOLO model
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# # Load COCO class labels
# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")

# # Get the output layer names from YOLO
# layer_names = net.getUnconnectedOutLayersNames()

# # Open webcam
# #cap = cv2.VideoCapture(1)  # 0 = default webcam 1 = IPhone webcam
# # Open video file
# cap = cv2.VideoCapture("video1.mp4")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     height, width = frame.shape[:2]

#     # Convert frame to blob for YOLO
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)

#     # Forward pass through YOLO
#     outputs = net.forward(layer_names)

#     boxes, confidences, class_ids = [], [], []

#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:  # Confidence threshold
#                 center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
#                 x, y = center_x - w // 2, center_y - h // 2

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Draw bounding boxes and labels
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
#             color = (0, 255, 0)  # Green
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Show the output
#     cv2.imshow("YOLO Real-Time Detection", frame)

#     if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mss

# def get_display_info():
#     with mss.mss() as sct:
#         # List all monitors
#         for idx, monitor in enumerate(sct.monitors[1:], 1):  # Skip primary monitor
#             print(f"Monitor {idx}: {monitor}")
#         return sct.monitors

# def setup_monitor(monitor_number=2):  # 2 for second display
#     with mss.mss() as sct:
#         if monitor_number >= len(sct.monitors):
#             raise ValueError("Monitor number not available")
#         return sct.monitors[monitor_number]

# # Load YOLO model
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")

# layer_names = net.getUnconnectedOutLayersNames()

# # Get second monitor
# monitor = setup_monitor(2)  # Use 2 for second display

# with mss.mss() as sct:
#     while True:
#         frame = np.array(sct.grab(monitor))
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR (OpenCV format)

#         height, width = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#         net.setInput(blob)
#         outputs = net.forward(layer_names)

#         boxes, confidences, class_ids = [], [], []

#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]

#                 if confidence > 0.5:
#                     center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
#                     x, y = center_x - w // 2, center_y - h // 2
#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#         if len(indexes) > 0:
#             for i in indexes.flatten():
#                 x, y, w, h = boxes[i]
#                 label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
#                 color = (0, 255, 0)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         cv2.imshow("YOLO Screen Detection", frame)

#         if cv2.waitKey(1) == 27:  # Press ESC to exit
#             break

# cv2.destroyAllWindows()

import cv2
import numpy as np
import mss
from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionOnScreenOnly

def get_chrome_bounds():
    """Find the position of the Google Chrome window."""
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    
    for window in window_list:
        if "Google Chrome" in window.get("kCGWindowOwnerName", ""):
            bounds = window["kCGWindowBounds"]
            return bounds["X"], bounds["Y"], bounds["Width"], bounds["Height"]

    return None

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getUnconnectedOutLayersNames()

with mss.mss() as sct:
    while True:
        bounds = get_chrome_bounds()
        if not bounds:
            print("Google Chrome window not found!")
            break

        x, y, width, height = bounds
        screen = np.array(sct.grab({"top": y, "left": x, "width": width, "height": height}))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

        # YOLO Object Detection
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(layer_names)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                    x1, y1 = center_x - w // 2, center_y - h // 2
                    boxes.append([x1, y1, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the result
        cv2.imshow("Google Chrome - YOLO Object Detection", frame)

        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

cv2.destroyAllWindows()