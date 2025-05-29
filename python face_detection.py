# Save as face_detection_opencv.py
import cv2
from cvzone.FaceDetectionModule import FaceDetector
import time

# Initialize camera - try different indices if needed
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera. Trying different camera index...")
    cap = cv2.VideoCapture(1)  # Try camera index 1
    if not cap.isOpened():
        print("Error: No camera found!")
        exit()

print("Camera opened successfully!")

# Initialize face detector
detector = FaceDetector(minDetectionCon=0.5)

print("Starting face detection... Press 'q' to quit")

frame_count = 0
start_time = time.time()

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Detect faces
        frame, bboxs = detector.findFaces(frame)
        
        # Add some info to the frame
        if bboxs:
            # Draw face count on screen
            cv2.putText(frame, f'Faces: {len(bboxs)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Faces detected: {len(bboxs)}")
        else:
            cv2.putText(frame, 'No faces detected', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Detection - Press Q to quit', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nStopping face detection...")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")