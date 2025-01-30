import cv2
from hands import HandDetector

def calculate_water_needed(coffee_count):
    return (coffee_count * 250) / 1000  # Convert to liters

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        results = detector.process_frame(frame)
        
        coffee_count = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                detector.draw_landmarks(frame, hand_landmarks)
                coffee_count = detector.count_raised_fingers(hand_landmarks)
        
        water_needed = calculate_water_needed(coffee_count)
        
        cv2.putText(frame, f'Coffee cups: {coffee_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Water needed: {water_needed:.2f}L', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Coffee Counter', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()