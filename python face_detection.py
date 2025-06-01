# Face Recognition and Matching System for Raspberry Pi - STABILITY OPTIMIZED VERSION
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
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StableFaceRecognitionSystem:
    def __init__(self, width=320, height=240, fps=10):  # Reduced resolution and FPS
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe_name = '/tmp/video_pipe'
        self.detector = FaceDetector(minDetectionCon=0.6)  # Higher confidence threshold
        self.cap = None
        self.rpicam_process = None
        
        # Simplified threading - single worker only
        self.processing_pool = ThreadPoolExecutor(max_workers=1)
        self.processing_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Aggressive frame processing optimization
        self.skip_frames = 5  # Process every 6th frame
        self.frame_skip_counter = 0
        
        # Face recognition data
        self.known_faces = {}
        self.known_names = []
        self.known_encodings = []
        self.faces_data_file = 'known_faces.pkl'
        
        # Conservative matching parameters
        self.match_threshold = 0.5  # Lower threshold for better matches
        self.recognition_enabled = False
        
        # UI state management
        self.adding_face_mode = False
        self.pending_face_frame = None
        self.face_name_buffer = ""
        self.input_mode = False
        
        # Extended caching for performance
        self.last_recognition_result = []
        self.last_recognition_frame = 0
        self.recognition_cache_duration = 15  # Cache results longer
        
        # Memory management
        self.max_known_faces = 10  # Limit number of stored faces
        self.cleanup_counter = 0
        
        # Load existing face data
        self.load_known_faces()
        
        # Statistics
        self.frame_count = 0
        self.processed_frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # Single future for async processing
        self.current_future = None
        
        # Emergency memory management
        self.last_gc_time = time.time()
        self.gc_interval = 10  # Force garbage collection every 10 seconds

    def setup_camera_pipe(self):
        """Set up the named pipe for camera streaming with stability focus"""
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)
        
        os.mkfifo(self.pipe_name)
        
        # Start rpicam-vid process with minimal settings
        print("Starting camera process...")
        self.rpicam_process = subprocess.Popen([
            'rpicam-vid', 
            '-t', '0',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(self.fps),
            '--codec', 'mjpeg',
            '--nopreview',
            '--immediate',  # Start immediately without warm-up
            '-o', self.pipe_name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(3)  # Give more time for pipe to be ready
        
        print("Opening camera pipe...")
        self.cap = cv2.VideoCapture(self.pipe_name)
        
        if not self.cap.isOpened():
            raise Exception("Failed to open camera pipe!")
            
        # Minimal buffer to prevent memory buildup
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Camera pipe setup successful!")
        cv2.namedWindow('Face Recognition System', cv2.WINDOW_AUTOSIZE)
    
    def recognize_faces_ultra_safe(self, frame):
        """Ultra-conservative face recognition to prevent crashes"""
        try:
            if not self.known_encodings:
                return []

            # Very small frame for processing - critical for Pi stability
            small_frame = cv2.resize(frame, (80, 60))  # Very small resolution
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Most conservative face detection
            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                model="hog",  # Faster than CNN
                number_of_times_to_upsample=0  # No upsampling
            )
            
            if not face_locations:
                return []

            # Process only the first (largest) face to prevent overload
            face_locations = face_locations[:1]
            
            recognized_faces = []
            
            try:
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame,
                    known_face_locations=face_locations,
                    num_jitters=0,  # No jittering for stability
                    model="small"   # Use small model
                )
                
                if not face_encodings:
                    return []
                
                face_encoding = face_encodings[0]
                
                # Simple distance calculation
                distances = []
                for known_encoding in self.known_encodings:
                    distance = np.linalg.norm(
                        np.array(face_encoding, dtype=np.float32) - 
                        np.array(known_encoding, dtype=np.float32)
                    )
                    distances.append(distance)
                
                if distances:
                    best_match_idx = np.argmin(distances)
                    best_distance = distances[best_match_idx]
                    confidence = max(0, 1 - best_distance)
                    
                    if confidence > self.match_threshold:
                        name = self.known_names[best_match_idx]
                    else:
                        name = "Unknown"
                        confidence = 0
                    
                    # Scale coordinates back to original frame
                    top, right, bottom, left = face_locations[0]
                    scale_x = frame.shape[1] / small_frame.shape[1]
                    scale_y = frame.shape[0] / small_frame.shape[0]
                    
                    recognized_faces.append({
                        'name': name,
                        'confidence': confidence,
                        'location': (
                            int(left * scale_x),
                            int(top * scale_y),
                            int(right * scale_x),
                            int(bottom * scale_y)
                        )
                    })
                    
            except Exception as e:
                logger.error(f"Encoding error: {e}")
                return []
                
            return recognized_faces
            
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return []
        finally:
            # Force cleanup
            if 'rgb_small_frame' in locals():
                del rgb_small_frame
            if 'small_frame' in locals():
                del small_frame
    
    def process_recognition_async(self, frame, frame_number):
        """Process face recognition asynchronously with error handling"""
        try:
            result = self.recognize_faces_ultra_safe(frame)
            return frame_number, result
        except Exception as e:
            logger.error(f"Async recognition error: {e}")
            return frame_number, []
        finally:
            # Clean up frame reference
            del frame
    
    def force_garbage_collection(self):
        """Force garbage collection periodically"""
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            gc.collect()
            self.last_gc_time = current_time
    
    def save_known_faces(self):
        """Save known faces to file with size limit"""
        if len(self.known_names) > self.max_known_faces:
            # Remove oldest faces if we exceed limit
            excess = len(self.known_names) - self.max_known_faces
            self.known_names = self.known_names[excess:]
            self.known_encodings = self.known_encodings[excess:]
            logger.warning(f"Removed {excess} oldest faces to stay within limit")
        
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
                self.known_names = data['names'][:self.max_known_faces]  # Limit loaded faces
                self.known_encodings = data['encodings'][:self.max_known_faces]
                print(f"Loaded {len(self.known_names)} known faces from {self.faces_data_file}")
            except Exception as e:
                print(f"Error loading known faces: {e}")
                self.known_names = []
                self.known_encodings = []
        else:
            print("No existing face data found. Starting fresh.")
    
    def add_face_safe(self, frame, name):
        """Safely add a new face with memory management"""
        try:
            # Check if we're at the limit
            if len(self.known_names) >= self.max_known_faces:
                print(f"Cannot add more faces. Limit of {self.max_known_faces} reached.")
                print("Clear some faces first using 'c' command.")
                return False
            
            # Use smaller frame for encoding to save memory
            small_frame = cv2.resize(frame, (160, 120))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if len(face_locations) == 0:
                print("No face found in the image!")
                return False
            
            if len(face_locations) > 1:
                print("Multiple faces found! Please ensure only one face is visible.")
                return False
            
            face_encodings = face_recognition.face_encodings(
                rgb_frame, 
                face_locations, 
                num_jitters=1,
                model="small"
            )
            
            if not face_encodings:
                print("Could not encode the face!")
                return False
            
            self.known_encodings.append(face_encodings[0])
            self.known_names.append(name)
            self.save_known_faces()
            
            print(f"Face '{name}' added successfully! ({len(self.known_names)}/{self.max_known_faces})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False
        finally:
            # Clean up
            gc.collect()
    
    def draw_recognition_results(self, frame, recognized_faces):
        """Draw recognition results on the frame"""
        for face in recognized_faces:
            left, top, right, bottom = face['location']
            name = face['name']
            confidence = face['confidence']
            
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            label = f"{name}"
            if name != "Unknown" and confidence > 0:
                label += f" ({confidence:.2f})"
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    
    def compare_two_faces(self, recognized_faces):
        """Compare two faces and determine if they match"""
        if len(recognized_faces) == 2:
            face1, face2 = recognized_faces
            
            if (face1['name'] == face2['name'] and 
                face1['name'] != "Unknown" and 
                face1['confidence'] > 0.4 and 
                face2['confidence'] > 0.4):
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
                    if self.add_face_safe(self.pending_face_frame, name):
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
            if len(self.face_name_buffer) < 20:  # Limit name length
                self.face_name_buffer += chr(key)
    
    def run(self):
        """Main recognition loop with maximum stability"""
        try:
            self.setup_camera_pipe()
            
            print("\n=== STABLE Face Recognition System for Raspberry Pi ===")
            print("OPTIMIZED FOR STABILITY - Lower performance but crash-resistant")
            print("Controls:")
            print("  'q' - Quit")
            print("  'a' - Add current face (type name and press ENTER)")
            print("  'r' - Toggle recognition mode")
            print("  'c' - Clear all known faces")
            print("  's' - Save screenshot")
            print("  'l' - List known faces")
            print("  '+' - Decrease frame skipping (higher CPU usage)")
            print("  '-' - Increase frame skipping (lower CPU usage)")
            print(f"\nKnown faces: {len(self.known_names)}/{self.max_known_faces}")
            print(f"Frame skip: {self.skip_frames} (processing every {self.skip_frames + 1} frames)")
            print(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
            print("\nMake sure the OpenCV window has focus for key presses to work!")
            
            screenshot_count = 0
            recognized_faces = []
            
            while True:
                try:
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
                    
                    # Handle completed recognition future
                    if self.current_future and self.current_future.done():
                        try:
                            frame_num, result = self.current_future.result()
                            with self.processing_lock:
                                self.last_recognition_result = result
                                self.last_recognition_frame = frame_num
                            self.processed_frame_count += 1
                            self.current_future = None
                        except Exception as e:
                            logger.error(f"Recognition future error: {e}")
                            self.current_future = None
                    
                    # Start new recognition task if needed
                    self.frame_skip_counter += 1
                    should_process = (self.frame_skip_counter % (self.skip_frames + 1) == 0)
                    
                    if (should_process and 
                        self.recognition_enabled and 
                        len(self.known_encodings) > 0 and 
                        face_count > 0 and
                        self.current_future is None):  # Only one future at a time
                        
                        self.current_future = self.processing_pool.submit(
                            self.process_recognition_async, 
                            original_frame.copy(), 
                            self.frame_count
                        )
                    
                    # Use cached recognition results
                    if self.recognition_enabled:
                        with self.processing_lock:
                            if (self.last_recognition_result and 
                                self.frame_count - self.last_recognition_frame < self.recognition_cache_duration):
                                recognized_faces = self.last_recognition_result
                            elif not self.current_future:  # No active processing
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
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Display info overlays with smaller text
                    cv2.putText(frame_with_faces, f'Faces: {face_count}', (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    fps = self.calculate_fps()
                    cv2.putText(frame_with_faces, f'FPS: {fps:.1f}', (10, 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                    
                    status = "ON" if self.recognition_enabled else "OFF"
                    status_color = (0, 255, 0) if self.recognition_enabled else (0, 0, 255)
                    cv2.putText(frame_with_faces, f'Recognition: {status}', (10, 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
                    
                    cv2.putText(frame_with_faces, f'Known: {len(self.known_names)}/{self.max_known_faces}', (10, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    
                    # Processing status indicator
                    if self.current_future:
                        cv2.putText(frame_with_faces, 'PROCESSING...', (10, 105), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Display input mode status
                    if self.adding_face_mode:
                        if self.input_mode:
                            input_text = f"Name: {self.face_name_buffer}_"
                            cv2.putText(frame_with_faces, input_text, (10, 130), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame_with_faces, "Type name and press ENTER (ESC to cancel)", 
                                       (10, frame_with_faces.shape[0] - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        else:
                            cv2.putText(frame_with_faces, "Face captured! Now type the name...", 
                                       (10, 130), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Instructions
                    cv2.putText(frame_with_faces, "Press 'r' to toggle recognition, 'a' to add face", 
                               (10, frame_with_faces.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Show the frame
                    cv2.imshow('Face Recognition System', frame_with_faces)
                    
                    # Periodic garbage collection
                    self.force_garbage_collection()
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    if self.input_mode:
                        self.handle_text_input(key)
                    else:
                        if key == ord('q'):
                            break
                            
                        elif key == ord('a'):
                            if face_count == 1:
                                if len(self.known_names) < self.max_known_faces:
                                    print(f"Adding face mode activated. Face count: {face_count}")
                                    self.adding_face_mode = True
                                    self.input_mode = True
                                    self.pending_face_frame = original_frame.copy()
                                    self.face_name_buffer = ""
                                else:
                                    print(f"Cannot add more faces. Limit reached ({self.max_known_faces})")
                            else:
                                print(f"Please ensure exactly 1 face is visible (currently {face_count})")
                        
                        elif key == ord('r'):
                            self.recognition_enabled = not self.recognition_enabled
                            status = "enabled" if self.recognition_enabled else "disabled"
                            print(f"Recognition {status}")
                            if not self.recognition_enabled:
                                # Cancel any pending recognition
                                if self.current_future:
                                    self.current_future.cancel()
                                    self.current_future = None
                                recognized_faces = []
                        
                        elif key == ord('c'):
                            self.known_names.clear()
                            self.known_encodings.clear()
                            self.save_known_faces()
                            self.last_recognition_result = []
                            recognized_faces = []
                            print("All known faces cleared!")
                            gc.collect()  # Force cleanup after clearing
                        
                        elif key == ord('s'):
                            screenshot_count += 1
                            filename = f'face_recognition_{screenshot_count}.jpg'
                            cv2.imwrite(filename, frame_with_faces)
                            print(f"Screenshot saved as {filename}")
                        
                        elif key == ord('l'):
                            print(f"\nKnown faces ({len(self.known_names)}/{self.max_known_faces}):")
                            for i, name in enumerate(self.known_names):
                                print(f"  {i+1}. {name}")
                            print()
                        
                        elif key == ord('+'):
                            if self.skip_frames > 0:
                                self.skip_frames -= 1
                                print(f"Frame skip reduced to {self.skip_frames} (processing every {self.skip_frames + 1} frames)")
                        
                        elif key == ord('-'):
                            if self.skip_frames < 15:
                                self.skip_frames += 1
                                print(f"Frame skip increased to {self.skip_frames} (processing every {self.skip_frames + 1} frames)")
                
                except Exception as frame_error:
                    logger.error(f"Frame processing error: {frame_error}")
                    time.sleep(0.1)  # Brief pause before continuing
                    continue
                
        except KeyboardInterrupt:
            print("\nStopping face recognition...")
        
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Starting cleanup...")
        
        # Stop event to signal threads
        self.stop_event.set()
        
        # Cancel current future
        if self.current_future:
            self.current_future.cancel()
        
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
        
        # Final garbage collection
        gc.collect()
        
        print("Cleanup completed!")

# Main execution
if __name__ == "__main__":
    print("Starting STABLE Face Recognition System for Raspberry Pi")
    print("This version prioritizes stability over performance")
    print("If you still experience crashes, try reducing resolution further or increasing skip_frames")
    
    try:
        # Create and run the stable face recognition system
        recognition_system = StableFaceRecognitionSystem(width=320, height=240, fps=10)
        recognition_system.run()
    except Exception as e:
        print(f"System error: {e}")
        logger.error(f"System error: {e}")
    finally:
        print("System shutdown complete")