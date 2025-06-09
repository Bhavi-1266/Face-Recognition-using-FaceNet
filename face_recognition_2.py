# Face Recognition and Matching System for Raspberry Pi - Manual Training Version
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
import glob

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
        self.processing_pool = ThreadPoolExecutor(max_workers=1)
        self.processing_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Frame processing optimization
        self.skip_frames = 3  # Process every 3rd frame for recognition
        self.frame_skip_counter = 0
        
        # Face recognition data - modified for multiple encodings per person
        self.known_faces = {}  # person_name: [list of encodings]
        self.known_names = []  # List of unique names
        self.faces_data_file = 'known_faces.pkl'
        self.training_images_dir = 'training_images'
        
        # Matching parameters
        self.match_threshold = 0.6
        self.min_votes = 2  # Minimum votes needed for recognition
        self.recognition_enabled = False
        
        # Caching for performance
        self.last_recognition_result = []
        self.last_recognition_frame = 0
        self.recognition_cache_duration = 5  # frames
        
        # Create training directory if it doesn't exist
        if not os.path.exists(self.training_images_dir):
            os.makedirs(self.training_images_dir)
            print(f"Created training images directory: {self.training_images_dir}")
        
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

    def train_faces_from_directory(self):
        """Train faces from images in the training_images directory"""
        print("\n=== Training Face Recognition Model ===")
        
        # Clear existing data
        self.known_faces = {}
        self.known_names = []
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(self.training_images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.training_images_dir, ext.upper())))
        
        if not image_files:
            print(f"No training images found in {self.training_images_dir}")
            print("Please add images with format: PersonName_1.jpg, PersonName_2.jpg, etc.")
            return False
        
        print(f"Found {len(image_files)} training images")
        
        # Process each image
        for image_path in sorted(image_files):
            filename = os.path.basename(image_path)
            name_part = filename.split('.')[0]  # Remove extension
            
            # Extract person name (everything before the last underscore)
            if '_' in name_part:
                person_name = name_part.rsplit('_', 1)[0]
            else:
                person_name = name_part
            
            print(f"Processing {filename} for person: {person_name}")
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Error: Could not load {filename}")
                continue
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if len(face_encodings) == 0:
                print(f"  Warning: No face found in {filename}")
                continue
            elif len(face_encodings) > 1:
                print(f"  Warning: Multiple faces found in {filename}, using the first one")
            
            # Add encoding to the person's list
            encoding = face_encodings[0]
            
            if person_name not in self.known_faces:
                self.known_faces[person_name] = []
                self.known_names.append(person_name)
            
            self.known_faces[person_name].append(encoding)
            print(f"  Added encoding for {person_name} (total: {len(self.known_faces[person_name])})")
        
        # Save the trained model
        self.save_known_faces()
        
        # Print training summary
        print("\n=== Training Summary ===")
        total_encodings = 0
        for name, encodings in self.known_faces.items():
            count = len(encodings)
            total_encodings += count
            print(f"  {name}: {count} encodings")
        
        print(f"Total: {len(self.known_names)} people, {total_encodings} encodings")
        print("Training completed successfully!\n")
        
        return True

    def setup_camera_pipe(self):
        """Set up the named pipe for camera streaming"""
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
        """Recognize faces using voting system with multiple encodings per person"""
        try:
            with self.processing_lock:
                if not self.known_faces:
                    return []

                # Ultra-lightweight processing
                small_frame = cv2.resize(frame, (160, 120))
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                # Conservative face detection
                face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    model="hog",
                    number_of_times_to_upsample=0
                )
                
                if not face_locations:
                    return []

                # Process faces
                recognized_faces = []
                for top, right, bottom, left in face_locations:
                    try:
                        face_encoding = face_recognition.face_encodings(
                            rgb_small_frame,
                            known_face_locations=[(top, right, bottom, left)],
                            num_jitters=0
                        )[0]
                        
                        # Voting system: compare against all encodings for each person
                        person_votes = {}
                        person_confidences = {}
                        
                        for person_name, person_encodings in self.known_faces.items():
                            votes = 0
                            confidences = []
                            
                            for encoding in person_encodings:
                                distance = face_recognition.face_distance([encoding], face_encoding)[0]
                                confidence = 1 - distance
                                
                                if confidence > self.match_threshold:
                                    votes += 1
                                    confidences.append(confidence)
                            
                            if votes >= self.min_votes:
                                person_votes[person_name] = votes
                                person_confidences[person_name] = np.mean(confidences)
                        
                        # Determine best match
                        if person_votes:
                            # Find person with most votes, break ties with highest confidence
                            best_person = max(person_votes.keys(), 
                                            key=lambda x: (person_votes[x], person_confidences[x]))
                            
                            name = best_person
                            confidence = person_confidences[best_person]
                            votes = person_votes[best_person]
                        else:
                            name = "Unknown"
                            confidence = 0
                            votes = 0
                        
                        # Scale coordinates back
                        scale_x = frame.shape[1] / small_frame.shape[1]
                        scale_y = frame.shape[0] / small_frame.shape[0]
                        
                        recognized_faces.append({
                            'name': name,
                            'confidence': confidence,
                            'votes': votes,
                            'location': (
                                int(left * scale_x),
                                int(top * scale_y),
                                int(right * scale_x),
                                int(bottom * scale_y)
                            )
                        })
                        
                    except Exception as e:
                        print(f"Face processing error: {e}")
                        continue
                        
                return recognized_faces
            
        except Exception as e:
            print(f"Recognition crash: {e}")
            return []
    
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
            'faces': self.known_faces,
            'names': self.known_names
        }
        with open(self.faces_data_file, 'wb') as f:
            pickle.dump(data, f)
        
        total_encodings = sum(len(encodings) for encodings in self.known_faces.values())
        print(f"Saved {len(self.known_names)} people with {total_encodings} total encodings to {self.faces_data_file}")
    
    def load_known_faces(self):
        """Load known faces from file"""
        if os.path.exists(self.faces_data_file):
            try:
                with open(self.faces_data_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.known_faces = data.get('faces', {})
                self.known_names = data.get('names', [])
                
                total_encodings = sum(len(encodings) for encodings in self.known_faces.values())
                print(f"Loaded {len(self.known_names)} people with {total_encodings} total encodings from {self.faces_data_file}")
                
            except Exception as e:
                print(f"Error loading known faces: {e}")
                self.known_faces = {}
                self.known_names = []
        else:
            print("No existing face data found.")
    
    def draw_recognition_results(self, frame, recognized_faces):
        """Draw recognition results on the frame with vote information"""
        for face in recognized_faces:
            left, top, right, bottom = face['location']
            name = face['name']
            confidence = face['confidence']
            votes = face.get('votes', 0)
            
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            if name != "Unknown":
                label = f"{name} ({confidence:.2f}, {votes}v)"
            else:
                label = "Unknown"
            
            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            
            cv2.rectangle(frame, (left, bottom - text_height - 10), 
                         (left + text_width + 12, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
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
    
    def run(self):
        """Main recognition loop"""
        try:
            self.setup_camera_pipe()
            
            print("\n=== Face Recognition System with Manual Training ===")
            print("Controls:")
            print("  'q' - Quit")
            print("  'r' - Toggle recognition mode")
            print("  't' - Retrain from images")
            print("  's' - Save screenshot")
            print("  'l' - List known faces")
            print("  '+' - Decrease frame skipping (higher CPU usage)")
            print("  '-' - Increase frame skipping (lower CPU usage)")
            print(f"\nKnown people: {len(self.known_names)}")
            if self.known_faces:
                total_encodings = sum(len(encodings) for encodings in self.known_faces.values())
                print(f"Total encodings: {total_encodings}")
            print(f"Frame skip: {self.skip_frames} (processing every {self.skip_frames + 1} frames)")
            print(f"Training images directory: {self.training_images_dir}")
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
                    len(self.known_faces) > 0 and 
                    face_count > 0 and
                    len(self.pending_futures) < 2):
                    
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
                
                cv2.putText(frame_with_faces, f'People: {len(self.known_names)}', (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(frame_with_faces, f'Skip: {self.skip_frames}', (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                
                cv2.putText(frame_with_faces, f'Processed: {self.processed_frame_count}', (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Instructions
                cv2.putText(frame_with_faces, "Press 't' to retrain, 'r' to toggle recognition", 
                           (10, frame_with_faces.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show the frame
                cv2.imshow('Face Recognition System', frame_with_faces)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord('t'):
                    print("Retraining model...")
                    self.train_faces_from_directory()
                
                elif key == ord('r'):
                    self.recognition_enabled = not self.recognition_enabled
                    status = "enabled" if self.recognition_enabled else "disabled"
                    print(f"Recognition {status}")
                
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f'face_recognition_{screenshot_count}.jpg'
                    cv2.imwrite(filename, frame_with_faces)
                    print(f"Screenshot saved as {filename}")
                
                elif key == ord('l'):
                    print(f"\nKnown people ({len(self.known_names)}):")
                    for name in sorted(self.known_names):
                        encoding_count = len(self.known_faces[name])
                        print(f"  {name}: {encoding_count} encodings")
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
    print("=== Face Recognition Training Setup ===")
    print("\nTo use this system:")
    print("1. Create a 'training_images' directory")
    print("2. Add multiple images for each person with naming format:")
    print("   PersonName_1.jpg, PersonName_2.jpg, PersonName_3.jpg, etc.")
    print("   Example: John_1.jpg, John_2.jpg, Mary_1.jpg, Mary_2.jpg")
    print("3. Press 't' during runtime to train the model")
    print("4. Press 'r' to enable recognition")
    print("\nStarting system...")
    
    # Create and run the face recognition system
    recognition_system = ThreadedFaceRecognitionSystem(width=640, height=480, fps=15)
    recognition_system.run()