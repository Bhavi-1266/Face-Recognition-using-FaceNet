# Face Recognition and Matching System for Raspberry Pi - FIXED VERSION
import cv2
import subprocess
import time
import os
import numpy as np
import pickle
from cvzone.FaceDetectionModule import FaceDetector
import face_recognition
import threading
import sys

class FaceRecognitionSystem:
    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe_name = '/tmp/video_pipe'
        self.detector = FaceDetector(minDetectionCon=0.5)
        self.cap = None
        self.rpicam_process = None
        
        # Face recognition data
        self.known_faces = {}  # Dictionary to store known face encodings
        self.known_names = []  # List of known face names
        self.known_encodings = []  # List of known face encodings
        self.faces_data_file = 'known_faces.pkl'
        
        # Matching parameters
        self.match_threshold = 0.6  # Lower = more strict matching
        self.recognition_enabled = False
        
        # UI state management
        self.adding_face_mode = False
        self.pending_face_frame = None
        self.face_name_buffer = ""
        self.input_mode = False
        
        # Load existing face data
        self.load_known_faces()
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        
    def setup_camera_pipe(self):
        """Set up the named pipe for camera streaming"""
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)
        
        os.mkfifo(self.pipe_name)
        
        self.rpicam_process = subprocess.Popen([
            'rpicam-vid', 
            '-t', '0',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(self.fps),
            '--codec', 'mjpeg',
            '-o', self.pipe_name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(2)
        self.cap = cv2.VideoCapture(self.pipe_name)
        
        if not self.cap.isOpened():
            raise Exception("Failed to open camera pipe!")
            
        print("Camera pipe setup successful!")
        
        # Make sure OpenCV window has focus
        cv2.namedWindow('Face Recognition System', cv2.WINDOW_AUTOSIZE)
    
    def save_known_faces(self):
        """Save known faces to file"""
        data = {
            'names': self.known_names,
            'encodings': self.known_encodings
        }
        with open(self.faces_data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.known_names)} known faces to {self.faces_data_file}")
    
    def load_known_faces(self):
        """Load known faces from file"""
        if os.path.exists(self.faces_data_file):
            try:
                with open(self.faces_data_file, 'rb') as f:
                    data = pickle.load(f)
                self.known_names = data['names']
                self.known_encodings = data['encodings']
                print(f"Loaded {len(self.known_names)} known faces from {self.faces_data_file}")
            except Exception as e:
                print(f"Error loading known faces: {e}")
                self.known_names = []
                self.known_encodings = []
        else:
            print("No existing face data found. Starting fresh.")
    
    def add_face(self, frame, name):
        """Add a new face to the known faces database"""
        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) == 0:
            print("No face found in the image!")
            return False
        
        if len(face_encodings) > 1:
            print("Multiple faces found! Please ensure only one face is visible.")
            return False
        
        # Add the face encoding
        self.known_encodings.append(face_encodings[0])
        self.known_names.append(name)
        
        # Save to file
        self.save_known_faces()
        
        print(f"Face '{name}' added successfully!")
        return True
    
    def recognize_faces(self, frame):
        """Recognize faces in the current frame"""
        if len(self.known_encodings) == 0:
            return []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.match_threshold)
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            
            name = "Unknown"
            confidence = 0
            
            if True in matches:
                # Find the best match
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - distances[best_match_index]
            
            recognized_faces.append({
                'name': name,
                'confidence': confidence,
                'location': (left, top, right, bottom)
            })
        
        return recognized_faces
    
    def draw_recognition_results(self, frame, recognized_faces):
        """Draw recognition results on the frame"""
        for face in recognized_faces:
            left, top, right, bottom = face['location']
            name = face['name']
            confidence = face['confidence']
            
            # Choose color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw name and confidence
            label = f"{name}"
            if name != "Unknown":
                label += f" ({confidence:.2f})"
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    def compare_two_faces(self, recognized_faces):
        """Compare two faces and determine if they match"""
        if len(recognized_faces) == 2:
            face1, face2 = recognized_faces
            
            # If both faces are recognized as the same person
            if (face1['name'] == face2['name'] and 
                face1['name'] != "Unknown" and 
                face1['confidence'] > 0.5 and 
                face2['confidence'] > 0.5):
                return True, f"MATCH: Both faces are {face1['name']}"
            
            # If both are unknown, we can't determine
            elif face1['name'] == "Unknown" and face2['name'] == "Unknown":
                return None, "Both faces are unknown - cannot determine match"
            
            # If they're different known people
            elif (face1['name'] != face2['name'] and 
                  face1['name'] != "Unknown" and 
                  face2['name'] != "Unknown"):
                return False, f"NO MATCH: {face1['name']} vs {face2['name']}"
            
            else:
                return None, "Cannot determine - one face unknown"
        
        return None, ""
    
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
    
    def handle_text_input(self, key):
        """Handle text input for face naming"""
        if key == 13:  # Enter key
            if self.face_name_buffer.strip():
                name = self.face_name_buffer.strip()
                if self.pending_face_frame is not None:
                    if self.add_face(self.pending_face_frame, name):
                        print(f"Face '{name}' added successfully!")
                    else:
                        print("Failed to add face. Try again.")
                else:
                    print("No face frame stored. Try again.")
                
                # Reset input mode
                self.adding_face_mode = False
                self.input_mode = False
                self.face_name_buffer = ""
                self.pending_face_frame = None
            
        elif key == 27:  # Escape key
            print("Face adding cancelled.")
            self.adding_face_mode = False
            self.input_mode = False
            self.face_name_buffer = ""
            self.pending_face_frame = None
            
        elif key == 8:  # Backspace
            if self.face_name_buffer:
                self.face_name_buffer = self.face_name_buffer[:-1]
                
        elif 32 <= key <= 126:  # Printable characters
            self.face_name_buffer += chr(key)
    
    def run(self):
        """Main recognition loop"""
        try:
            self.setup_camera_pipe()
            
            print("\n=== Face Recognition System ===")
            print("Controls:")
            print("  'q' - Quit")
            print("  'a' - Add current face (type name and press ENTER)")
            print("  'r' - Toggle recognition mode")
            print("  'c' - Clear all known faces")
            print("  's' - Save screenshot")
            print("  'l' - List known faces")
            print(f"\nKnown faces: {len(self.known_names)}")
            print("\nMake sure the OpenCV window has focus for key presses to work!")
            
            fps = 0
            screenshot_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                original_frame = frame.copy()
                
                # Basic face detection
                frame, bboxs = self.detector.findFaces(frame)
                
                # Face recognition if enabled
                recognized_faces = []
                if self.recognition_enabled and len(self.known_encodings) > 0:
                    recognized_faces = self.recognize_faces(original_frame)
                    self.draw_recognition_results(frame, recognized_faces)
                
                # Compare faces if exactly 2 are detected
                match_result = ""
                if len(recognized_faces) == 2:
                    is_match, match_message = self.compare_two_faces(recognized_faces)
                    if is_match is not None:
                        match_result = match_message
                        color = (0, 255, 0) if is_match else (0, 0, 255)
                        cv2.putText(frame, match_result, (10, 180), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display info
                face_count = len(bboxs) if bboxs else 0
                cv2.putText(frame, f'Faces: {face_count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display FPS
                current_fps = self.calculate_fps()
                if current_fps:
                    fps = current_fps
                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Display recognition status
                status = "ON" if self.recognition_enabled else "OFF"
                cv2.putText(frame, f'Recognition: {status}', (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display known faces count
                cv2.putText(frame, f'Known: {len(self.known_names)}', (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display input mode status
                if self.adding_face_mode:
                    if self.input_mode:
                        input_text = f"Name: {self.face_name_buffer}_"
                        cv2.putText(frame, input_text, (10, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, "Type name and press ENTER (ESC to cancel)", 
                                   (10, frame.shape[0] - 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Face captured! Now type the name...", 
                                   (10, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Instructions
                cv2.putText(frame, "Press 'a' to add face, 'r' to toggle recognition", 
                           (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Face Recognition System', frame)
                
                # Handle key presses - increased wait time for better key detection
                key = cv2.waitKey(30) & 0xFF
                
                if self.input_mode:
                    # Handle text input mode
                    self.handle_text_input(key)
                    
                else:
                    # Handle normal command keys
                    if key == ord('q'):
                        break
                        
                    elif key == ord('a'):
                        if face_count == 1:
                            print(f"Adding face mode activated. Face count: {face_count}")
                            self.adding_face_mode = True
                            self.input_mode = True
                            self.pending_face_frame = original_frame.copy()
                            self.face_name_buffer = ""
                        else:
                            print(f"Please ensure exactly 1 face is visible (currently {face_count})")
                    
                    elif key == ord('r'):
                        self.recognition_enabled = not self.recognition_enabled
                        status = "enabled" if self.recognition_enabled else "disabled"
                        print(f"Recognition {status}")
                    
                    elif key == ord('c'):
                        self.known_names.clear()
                        self.known_encodings.clear()
                        self.save_known_faces()
                        print("All known faces cleared!")
                    
                    elif key == ord('s'):
                        screenshot_count += 1
                        filename = f'face_recognition_{screenshot_count}.jpg'
                        cv2.imwrite(filename, frame)
                        print(f"Screenshot saved as {filename}")
                    
                    elif key == ord('l'):
                        print(f"\nKnown faces ({len(self.known_names)}):")
                        for i, name in enumerate(self.known_names):
                            print(f"  {i+1}. {name}")
                        print()  # Empty line for readability
                    
        except KeyboardInterrupt:
            print("\nStopping face recognition...")
        
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
        
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)
        
        print("Cleanup completed!")

# Main execution
if __name__ == "__main__":
    # Create and run the face recognition system
    recognition_system = FaceRecognitionSystem(width=640, height=480, fps=15)
    recognition_system.run()