import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

detector = FaceDetector(minDetectionCon=0.5)

print("Starting face detection... Press 'q' to quit")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        frame, bboxs = detector.findFaces(frame)
        
        if bboxs:
            print(f"Faces detected: {len(bboxs)}")
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nStopping face detection...")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()