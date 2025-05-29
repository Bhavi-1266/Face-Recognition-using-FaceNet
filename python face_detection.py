# Save as face_detection_simple.py
import cv2
import subprocess
import time
from cvzone.FaceDetectionModule import FaceDetector

# Create a named pipe for video streaming
import os
pipe_name = '/tmp/video_pipe'

# Remove pipe if it exists
if os.path.exists(pipe_name):
    os.remove(pipe_name)

# Create named pipe
os.mkfifo(pipe_name)

# Start rpicam-vid to stream to the pipe
subprocess.Popen([
    'rpicam-vid', 
    '-t', '0',
    '--width', '640',
    '--height', '480',
    '--framerate', '15',
    '--codec', 'mjpeg',
    '-o', pipe_name
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Wait a moment for the pipe to be ready
time.sleep(2)

# Open the pipe with OpenCV
cap = cv2.VideoCapture(pipe_name)

if not cap.isOpened():
    print("Failed to open camera pipe!")
    exit()

# Initialize face detector
detector = FaceDetector(minDetectionCon=0.5)

print("Starting face detection... Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received")
            time.sleep(0.1)
            continue
        
        # Detect faces
        frame, bboxs = detector.findFaces(frame)
        
        # Add info
        if bboxs:
            cv2.putText(frame, f'Faces: {len(bboxs)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    # Clean up pipe
    if os.path.exists(pipe_name):
        os.remove(pipe_name)
