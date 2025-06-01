# Face Recognition and Matching System for Raspberry Pi - FIXED THREADED VERSION
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
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import collections

class ThreadedFaceRecognitionSystem:
    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe_name = '/tmp/video_pipe'
        self.detector = FaceDetector(minDetectionCon=0.5)
        self.cap = None
        self.rpicam_process = None
        
        # Threading components
        self.processing_pool = ThreadPoolExecutor(max_workers=2)
        self.processing_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Frame processing optimization
        self.skip_frames = 2  # Process every 3rd frame for recognition
        self.frame_skip_counter = 0
        
        # Face recognition data
        self.known_faces = {}
        self.known_names = []
        self.known_encodings = []
        self.faces_data_file = 'known_faces.pkl'
        
        # Matching parameters
        self.match_threshold = 0.6
        self.recognition_enabled = False
        
        # UI state management
        self.adding_face_mode = False
        self.pending_face_frame = None
        self.face_name_buffer = ""
        self.input_mode = False
        
        # Caching for performance
        self.last_recognition_result = []
        self.last_recognition_frame = 0
        self.recognition_cache_duration = 5  # frames
        
        # Load existing face data
        self.load_known_faces()
        
        # Statistics
        self.frame_count = 0
        self.processed_frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # Async processing storage
        self.pending_futures = []
        
    def disable_camera_services(self):
        """Disable interfering camera services"""
        try:
            print("Stopping interfering camera services...")
            
            # Kill any existing camera processes
            subprocess.run(['sudo', 'pkill', '-f', 'rpicam'], stderr=subprocess.DEVNULL)
            subprocess.run(['sudo', 'pkill', '-f', 'libcamera'], stderr=subprocess.DEVNULL)
            subprocess.run(['sudo', 'pkill', '-f', 'vcgencmd'], stderr=subprocess.DEVNULL)
            
            # Stop camera-related services
            services_to_stop = [
                'rpicam-hello.service',
                'camera.service',
                'libcamera.service'
            ]
            
            for service in services_to_stop:
                try:
                    subprocess.run(['sudo', 'systemctl', 'stop', service], 
                                 stderr=subprocess.DEVNULL, timeout=5)
                except:
                    pass
            
            time.sleep(1)
            print("Camera services stopped.")
            
        except Exception as e:
            print(f"Warning: Could not stop camera services: {e}")
        
    def setup_camera_pipe(self):
        """Set up the named pipe for camera streaming"""
        # First disable interfering services
        self.disable_camera_services()
        
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)
        
        os.mkfifo(self.pipe_name)
        
        # Start rpicam-vid process
        print("Starting camera process...")
        self.rpicam_process = subprocess.Popen([
            'rpicam-vid', 
            '-t', '0',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(self.fps),
            '--codec', 'mjpeg',
            '--nopreview',  # Disable preview window
            '-o', self.pipe_name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(2)  # Give more time for pipe to be ready
        
        print("Opening camera pipe...")
        self.cap = cv2.VideoCapture(self.pipe_name)
        
        if not self.cap.isOpened():
            raise Exception("Failed to open camera pipe!")
            
        # Optimize capture settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Camera pipe setup successful!")
        cv2.namedWindow('Face Recognition System', cv2.WINDOW_AUTOSIZE)
    
    def recognize_faces_optimized(self, frame):
        """Optimized face recognition with reduced image size"""
        if len(self.known_encodings) == 0:
            return []
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings in smaller image
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            
            name = "Unknown"
            confidence = 0
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < self.match_threshold:
                    name = self.known_names[best_match_index]
                    confidence = 1 - distances[best_match_index]
            
            recognized_faces.append({
                'name': name,
                'confidence': confidence,
                'location': (left, top, right, bottom)
            })
        
        return recognized_faces
    
    def process_recognition_async(self, frame, frame_number):
        """Process face recognition asynchronously"""
        try:
            result = self.recognize_faces_optimized(frame)
            return frame_number, result
        except Exception as e:
            print(f"Recognition error: {e}")
            return frame_number, []
    
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) == 0:
            print("No face found in the image!")
            return False
        
        if len(face_encodings) > 1:
            print("Multiple faces found! Please ensure only one face is visible.")
            return False
        
        self.known_encodings.append(face_encodings[0])
        self.known_names.append(name)
        self.save_known_faces()
        
        print(f"Face '{name}' added successfully!")
        return True
    
    def draw_recognition_results(self, frame, recognized_faces):
        """Draw recognition results on the frame"""
        for face in recognized_faces:
            left, top, right, bottom = face['location']
            name = face['name']
            confidence = face['confidence']
            
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            label = f"{name}"
            if name != "Unknown":
                label += f" ({confidence:.2f})"
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    def compare_two_faces(self, recognized_faces):
        """Compare two faces and determine if they match"""
        if len(recognized_faces) == 2:
            face1, face2 = recognized_faces
            
            if (face1['name'] == face2['name'] and 
                face1['name'] != "Unknown" and 
                face1['confidence'] > 0.5 and 
                face2['confidence'] > 0.5):
                return True, f"MATCH: Both faces are {face1['name']}"
            
            elif face1['name'] == "Unknown" and face2['name'] == "Unknown":
                return None, "Both faces are unknown - cannot determine match"
            
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
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        return self.current_fps
    
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
        """Main recognition loop with optimized threading"""
        try:
            self.setup_camera_pipe()
            
            print("\n=== Optimized Face Recognition System ===")
            print("Controls:")
            print("  'q' - Quit")
            print("  'a' - Add current face (type name and press ENTER)")
            print("  'r' - Toggle recognition mode")
            print("  'c' - Clear all known faces")
            print("  's' - Save screenshot")
            print("  'l' - List known faces")
            print("  '+' - Decrease frame skipping (higher CPU usage)")
            print("  '-' - Increase frame skipping (lower CPU usage)")
            print(f"\nKnown faces: {len(self.known_names)}")
            print(f"Frame skip: {self.skip_frames} (processing every {self.skip_frames + 1} frames)")
            print("\nMake sure the OpenCV window has focus for key presses to work!")
            
            screenshot_count = 0
            recognized_faces = []
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                original_frame = frame.copy()
                
                # Basic face detection (always performed)
                frame_with_faces, bboxs = self.detector.findFaces(frame)
                face_count = len(bboxs) if bboxs else 0
                
                # Handle completed recognition futures
                completed_futures = []
                for i, future in enumerate(self.pending_futures):
                    if future.done():
                        try:
                            frame_num, result = future.result()
                            with self.processing_lock:
                                self.last_recognition_result = result
                                self.last_recognition_frame = frame_num
                            self.processed_frame_count += 1
                            completed_futures.append(i)
                        except Exception as e:
                            print(f"Recognition future error: {e}")
                            completed_futures.append(i)
                
                # Remove completed futures
                for i in reversed(completed_futures):
                    self.pending_futures.pop(i)
                
                # Start new recognition task if needed
                self.frame_skip_counter += 1
                should_process = (self.frame_skip_counter % (self.skip_frames + 1) == 0)
                
                if (should_process and 
                    self.recognition_enabled and 
                    len(self.known_encodings) > 0 and 
                    face_count > 0 and
                    len(self.pending_futures) < 2):  # Limit concurrent futures
                    
                    future = self.processing_pool.submit(
                        self.process_recognition_async, 
                        original_frame.copy(), 
                        self.frame_count
                    )
                    self.pending_futures.append(future)
                
                # Use cached recognition results
                if self.recognition_enabled:
                    with self.processing_lock:
                        if (self.last_recognition_result and 
                            self.frame_count - self.last_recognition_frame < self.recognition_cache_duration):
                            recognized_faces = self.last_recognition_result
                        else:
                            recognized_faces = []
                
                # Draw recognition results
                if recognized_faces:
                    self.draw_recognition_results(frame_with_faces, recognized_faces)
                    
                    # Compare faces if exactly 2 are detected
                    if len(recognized_faces) == 2:
                        is_match, match_message = self.compare_two_faces(recognized_faces)
                        if is_match is not None:
                            color = (0, 255, 0) if is_match else (0, 0, 255)
                            cv2.putText(frame_with_faces, match_message, (10, 200), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display info overlays
                cv2.putText(frame_with_faces, f'Faces: {face_count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                fps = self.calculate_fps()
                cv2.putText(frame_with_faces, f'FPS: {fps:.1f}', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                status = "ON" if self.recognition_enabled else "OFF"
                cv2.putText(frame_with_faces, f'Recognition: {status}', (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.putText(frame_with_faces, f'Known: {len(self.known_names)}', (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(frame_with_faces, f'Skip: {self.skip_frames}', (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                
                cv2.putText(frame_with_faces, f'Processed: {self.processed_frame_count}', (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Display input mode status
                if self.adding_face_mode:
                    if self.input_mode:
                        input_text = f"Name: {self.face_name_buffer}_"
                        cv2.putText(frame_with_faces, input_text, (10, 220), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame_with_faces, "Type name and press ENTER (ESC to cancel)", 
                                   (10, frame_with_faces.shape[0] - 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame_with_faces, "Face captured! Now type the name...", 
                                   (10, 220), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Instructions
                cv2.putText(frame_with_faces, "Press 'a' to add face, 'r' to toggle recognition", 
                           (10, frame_with_faces.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show the frame
                cv2.imshow('Face Recognition System', frame_with_faces)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if self.input_mode:
                    self.handle_text_input(key)
                else:
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
                        cv2.imwrite(filename, frame_with_faces)
                        print(f"Screenshot saved as {filename}")
                    
                    elif key == ord('l'):
                        print(f"\nKnown faces ({len(self.known_names)}):")
                        for i, name in enumerate(self.known_names):
                            print(f"  {i+1}. {name}")
                        print()
                    
                    elif key == ord('+'):
                        if self.skip_frames > 0:
                            self.skip_frames -= 1
                            print(f"Frame skip reduced to {self.skip_frames} (processing every {self.skip_frames + 1} frames)")
                    
                    elif key == ord('-'):
                        if self.skip_frames < 10:
                            self.skip_frames += 1
                            print(f"Frame skip increased to {self.skip_frames} (processing every {self.skip_frames + 1} frames)")
                
        except KeyboardInterrupt:
            print("\nStopping face recognition...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Starting cleanup...")
        
        # Stop event to signal threads
        self.stop_event.set()
        
        # Cancel pending futures
        for future in self.pending_futures:
            future.cancel()
        
        # Shutdown thread pool
        self.processing_pool.shutdown(wait=True)
        
        # Clean up OpenCV and camera
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
    # Create and run the optimized face recognition system
    recognition_system = ThreadedFaceRecognitionSystem(width=640, height=480, fps=15)
    recognition_system.run()