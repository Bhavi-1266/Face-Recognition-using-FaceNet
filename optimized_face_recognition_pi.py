# Face Recognition and Matching System for Raspberry Pi - FINAL STABLE VERSION
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
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    filename='face_recognition.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        self.skip_frames = 4  # Process every 5th frame
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

    def setup_camera_pipe(self):
        """Set up the named pipe for camera streaming"""
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)
        os.mkfifo(self.pipe_name)

        print("Starting camera process...")
        self.rpicam_process = subprocess.Popen([
            'rpicam-vid', 
            '-t', '0',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(self.fps),
            '--codec', 'mjpeg',
            '--nopreview',
            '-o', self.pipe_name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        print("Opening camera pipe...")
        self.cap = cv2.VideoCapture(self.pipe_name)
        if not self.cap.isOpened():
            raise Exception("Failed to open camera pipe!")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Camera pipe setup successful!")
        cv2.namedWindow('Face Recognition System', cv2.WINDOW_AUTOSIZE)

    def recognize_faces_optimized(self, frame):
        try:
            with self.processing_lock:
                if not self.known_encodings:
                    return []

                small_frame = cv2.resize(frame, (160, 120))
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    model="hog",
                    number_of_times_to_upsample=0
                )
                if not face_locations:
                    return []

                recognized_faces = []
                for top, right, bottom, left in face_locations:
                    try:
                        face_encoding = face_recognition.face_encodings(
                            rgb_small_frame,
                            known_face_locations=[(top, right, bottom, left)],
                            num_jitters=0
                        )[0]

                        distances = face_recognition.face_distance(
                            [np.array(x, dtype=np.float16) for x in self.known_encodings],
                            np.array(face_encoding, dtype=np.float16)
                        )

                        best_match_idx = np.argmin(distances)
                        confidence = 1 - distances[best_match_idx]
                        name = self.known_names[best_match_idx] if confidence > self.match_threshold else "Unknown"

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
                        logging.error(f"Face processing error: {e}")
                        continue
                return recognized_faces
        except Exception as e:
            logging.critical(f"Recognition crash: {e}", exc_info=True)
            return []

    def process_recognition_async(self, frame, frame_number):
        try:
            result = self.recognize_faces_optimized(frame.copy())
            return frame_number, result
        except Exception as e:
            logging.error(f"Async recognition failed: {e}")
            return frame_number, []

    def save_known_faces(self):
        data = {
            'names': self.known_names,
            'encodings': self.known_encodings
        }
        with open(self.faces_data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.known_names)} known faces")

    def load_known_faces(self):
        if os.path.exists(self.faces_data_file):
            try:
                with open(self.faces_data_file, 'rb') as f:
                    data = pickle.load(f)
                self.known_names = data['names']
                self.known_encodings = data['encodings']
                print(f"Loaded {len(self.known_names)} known faces")
            except Exception as e:
                print(f"Error loading known faces: {e}")
                self.known_names = []
                self.known_encodings = []
        else:
            print("No existing face data found")

    def add_face(self, frame, name):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if len(face_encodings) == 0:
            print("No face found in image")
            return False
        if len(face_encodings) > 1:
            print("Multiple faces detected")
            return False
        self.known_encodings.append(face_encodings[0])
        self.known_names.append(name)
        self.save_known_faces()
        print(f"Face '{name}' added successfully!")
        return True

    def draw_recognition_results(self, frame, recognized_faces):
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
        current_time = time.time()
        self.fps_counter += 1
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
        return self.current_fps

    def handle_text_input(self, key):
        if key == 13:
            ...
        elif key == 27:
            ...
        elif key == 8:
            ...
        elif 32 <= key <= 126:
            ...

    def run(self):
        try:
            self.setup_camera_pipe()
            print("=== Face Recognition System ===")
            print("Controls:")
            print("  'q' - Quit")
            print("  'a' - Add current face (type name and press ENTER)")
            print("  'r' - Toggle recognition mode")
            print("  'c' - Clear all known faces")
            print("  's' - Save screenshot")
            print("  'l' - List known faces")
            print("  '+' - Decrease frame skipping (higher CPU usage)")
            print("  '-' - Increase frame skipping (lower CPU usage)")

            screenshot_count = 0
            recognized_faces = []

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.1)
                    continue

                frame_with_faces, bboxs = self.detector.findFaces(frame)
                face_count = len(bboxs) if bboxs else 0

                completed_futures = []
                for i, future in enumerate(self.pending_futures):
                    if future.done():
                        try:
                            frame_num, result = future.result(timeout=1)
                            with self.processing_lock:
                                self.last_recognition_result = result
                                self.last_recognition_frame = frame_num
                            self.processed_frame_count += 1
                            completed_futures.append(i)
                        except Exception as e:
                            logging.error(f"Future error: {e}")
                            completed_futures.append(i)
                for i in reversed(completed_futures):
                    self.pending_futures.pop(i)

                self.frame_skip_counter += 1
                should_process = (self.frame_skip_counter % (self.skip_frames + 1) == 0)
                if should_process and self.recognition_enabled and len(self.known_encodings) > 0 and face_count > 0 and len(self.pending_futures) < 2:
                    future = self.processing_pool.submit(self.process_recognition_async, frame.copy(), self.frame_count)
                    future._start_time = time.time()
                    self.pending_futures.append(future)

                with self.processing_lock:
                    if self.last_recognition_result and self.frame_count - self.last_recognition_frame < self.recognition_cache_duration:
                        recognized_faces = self.last_recognition_result
                    else:
                        recognized_faces = []

                if recognized_faces:
                    self.draw_recognition_results(frame_with_faces, recognized_faces)

                fps = self.calculate_fps()
                cv2.putText(frame_with_faces, f'FPS: {fps:.1f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                status = "ON" if self.recognition_enabled else "OFF"
                cv2.putText(frame_with_faces, f'Recognition: {status}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('Face Recognition System', frame_with_faces)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.recognition_enabled = not self.recognition_enabled
                    logging.info(f"Recognition toggled: {self.recognition_enabled}")

        except Exception as e:
            logging.critical(f"Critical error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("Starting cleanup...")
        self.stop_event.set()
        for future in self.pending_futures:
            future.cancel()
        self.processing_pool.shutdown(wait=True)
        if self.cap:
            self.cap.release()
        if self.rpicam_process:
            self.rpicam_process.terminate()
            self.rpicam_process.wait()
        cv2.destroyAllWindows()
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)
        logging.info("Cleanup completed!")

if __name__ == "__main__":
    recognition_system = ThreadedFaceRecognitionSystem(width=640, height=480, fps=15)
    recognition_system.run()