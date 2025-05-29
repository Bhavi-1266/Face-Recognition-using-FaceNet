# Enhanced Face Detection with Raspberry Pi Camera
import cv2
import subprocess
import time
import os
from cvzone.FaceDetectionModule import FaceDetector

class RPiFaceDetector:
    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe_name = '/tmp/video_pipe'
        self.detector = FaceDetector(minDetectionCon=0.5)
        self.cap = None
        self.rpicam_process = None
        
        # Statistics
        self.frame_count = 0
        self.face_count_history = []
        self.start_time = time.time()
        
    def setup_camera_pipe(self):
        """Set up the named pipe for camera streaming"""
        # Remove pipe if it exists
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)
        
        # Create named pipe
        os.mkfifo(self.pipe_name)
        
        # Start rpicam-vid to stream to the pipe
        self.rpicam_process = subprocess.Popen([
            'rpicam-vid', 
            '-t', '0',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(self.fps),
            '--codec', 'mjpeg',
            '-o', self.pipe_name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for pipe to be ready
        time.sleep(2)
        
        # Open the pipe with OpenCV
        self.cap = cv2.VideoCapture(self.pipe_name)
        
        if not self.cap.isOpened():
            raise Exception("Failed to open camera pipe!")
            
        print("Camera pipe setup successful!")
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time
            return fps
        return None
    
    def draw_face_info(self, frame, bboxs):
        """Draw face detection information on frame"""
        face_count = len(bboxs) if bboxs else 0
        
        # Draw face count
        color = (0, 255, 0) if face_count > 0 else (0, 0, 255)
        cv2.putText(frame, f'Faces: {face_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw individual face info
        if bboxs:
            for i, bbox in enumerate(bboxs):
                x, y, w, h = bbox["bbox"]
                score = bbox["score"][0]
                
                # Draw confidence score near each face
                cv2.putText(frame, f'{score:.2f}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return face_count
    
    def draw_statistics(self, frame, fps=None):
        """Draw performance statistics"""
        if fps:
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Average faces over last 30 frames
        if len(self.face_count_history) > 0:
            avg_faces = sum(self.face_count_history[-30:]) / min(30, len(self.face_count_history))
            cv2.putText(frame, f'Avg Faces: {avg_faces:.1f}', (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def run(self):
        """Main detection loop"""
        try:
            self.setup_camera_pipe()
            
            print("Starting enhanced face detection...")
            print("Controls:")
            print("  'q' - Quit")
            print("  's' - Save screenshot")
            print("  'r' - Reset statistics")
            
            fps = 0
            screenshot_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("No frame received, waiting...")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Detect faces
                frame, bboxs = self.detector.findFaces(frame)
                
                # Draw face information
                face_count = self.draw_face_info(frame, bboxs)
                self.face_count_history.append(face_count)
                
                # Calculate and display FPS
                current_fps = self.calculate_fps()
                if current_fps:
                    fps = current_fps
                
                self.draw_statistics(frame, fps)
                
                # Draw instructions
                cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                           (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Enhanced Face Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f'face_detection_screenshot_{screenshot_count}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
                elif key == ord('r'):
                    self.face_count_history.clear()
                    print("Statistics reset")
                    
        except KeyboardInterrupt:
            print("\nStopping face detection...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        
        if self.rpicam_process:
            self.rpicam_process.terminate()
            self.rpicam_process.wait()
        
        cv2.destroyAllWindows()
        
        # Clean up pipe
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)
        
        print("Cleanup completed!")

# Main execution
if __name__ == "__main__":
    # Create and run the face detector
    detector = RPiFaceDetector(width=640, height=480, fps=15)
    detector.run()