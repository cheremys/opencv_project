import cv2
import mediapipe as mp
import time

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

def count_raised_fingers(hand_landmarks):
    # List of finger tip IDs
    finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    count = 0
    
    if hand_landmarks:
        # Check each finger
        for tip_id in finger_tips[1:]:  # Excluding thumb
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                count += 1
    
    return count

def calculate_water_needed(coffee_count):
    # For each cup of coffee (125ml), recommend 250ml of water
    water_ml = coffee_count * 250
    return water_ml / 1000  # Convert to liters

while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    coffee_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            coffee_count = count_raised_fingers(hand_landmarks)
    
    # Calculate water needed
    water_needed = calculate_water_needed(coffee_count)
    
    # Display information
    cv2.putText(frame, f'Coffee cups: {coffee_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Water needed: {water_needed:.2f}L', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Coffee Counter', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()