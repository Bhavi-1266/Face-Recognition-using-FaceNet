#!/usr/bin/env python3
"""
Face Detection and Recognition System for Raspberry Pi 5
Using SSD MobileNet for face detection and MobileFaceNet for recognition
Optimized for edge computing with minimal dependencies
"""

import cv2
import numpy as np
import os
import pickle
import argparse
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDetector:
    """SSD MobileNet based face detector optimized for Raspberry Pi"""
    
    def __init__(self, model_path="models/", confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load SSD MobileNet model for face detection"""
        try:
            # Download these files to your models/ directory:
            # https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
            prototxt_path = os.path.join(model_path, "deploy.prototxt")
            model_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
            
            if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
                logger.error("Model files not found. Please download SSD face detection model.")
                logger.info("Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector")
                return False
            
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            logger.info("SSD Face Detection model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading face detection model: {e}")
            return False
    
    def detect_faces(self, image):
        """Detect faces in image using SSD"""
        if self.net is None:
            return []
        
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype("int")
                
                # Ensure coordinates are within image bounds
                x, y = max(0, x), max(0, y)
                x1, y1 = min(w, x1), min(h, y1)
                
                faces.append({
                    'box': (x, y, x1-x, y1-y),
                    'confidence': confidence,
                    'landmarks': None  # SSD doesn't provide landmarks
                })
        
        return faces

class MobileFaceNet:
    """MobileFaceNet for face recognition - lightweight version"""
    
    def __init__(self, model_path="models/mobilefacenet.onnx"):
        self.model_path = model_path
        self.net = None
        self.load_model()
    
    def load_model(self):
        """Load MobileFaceNet ONNX model"""
        try:
            if os.path.exists(self.model_path):
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                logger.info("MobileFaceNet model loaded successfully")
                return True
            else:
                logger.warning("MobileFaceNet model not found. Using basic feature extraction.")
                return False
        except Exception as e:
            logger.error(f"Error loading MobileFaceNet model: {e}")
            return False
    
    def preprocess_face(self, face_img):
        """Preprocess face image for feature extraction"""
        # Resize to 112x112 (MobileFaceNet input size)
        face_resized = cv2.resize(face_img, (112, 112))
        
        # Normalize
        face_normalized = (face_resized - 127.5) / 128.0
        
        # Convert to blob
        blob = cv2.dnn.blobFromImage(face_normalized, 1.0, (112, 112), (0, 0, 0), swapRB=True)
        return blob
    
    def extract_features(self, face_img):
        """Extract face features using MobileFaceNet"""
        if self.net is None:
            # Fallback to basic feature extraction using OpenCV
            return self.extract_basic_features(face_img)
        
        try:
            blob = self.preprocess_face(face_img)
            self.net.setInput(blob)
            features = self.net.forward()
            return features.flatten()
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return self.extract_basic_features(face_img)
    
    def extract_basic_features(self, face_img):
        """Basic feature extraction fallback using histogram and LBP"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        
        # Resize to standard size
        gray = cv2.resize(gray, (64, 64))
        
        # Extract histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten()
        
        # Extract LBP features (simplified)
        lbp_features = []
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                binary = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if gray[i+di, j+dj] >= center:
                            binary = binary * 2 + 1
                        else:
                            binary = binary * 2
                lbp_features.append(binary % 256)
        
        # Combine features
        features = np.concatenate([hist, np.histogram(lbp_features, bins=32)[0]])
        return features / np.linalg.norm(features)  # Normalize

class FaceRecognizer:
    """Face Recognition System combining detection and recognition"""
    
    def __init__(self, model_path="models/"):
        self.detector = FaceDetector(model_path)
        self.recognizer = MobileFaceNet(os.path.join(model_path, "mobilefacenet.onnx"))
        self.known_faces = {}
        self.face_database_path = "face_database.pkl"
        self.load_database()
    
    def load_database(self):
        """Load known faces database"""
        if os.path.exists(self.face_database_path):
            try:
                with open(self.face_database_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                logger.error(f"Error loading face database: {e}")
                self.known_faces = {}
    
    def save_database(self):
        """Save known faces database"""
        try:
            with open(self.face_database_path, 'wb') as f:
                pickle.dump(self.known_faces, f)
            logger.info("Face database saved successfully")
        except Exception as e:
            logger.error(f"Error saving face database: {e}")
    
    def add_face(self, image, name, face_box=None):
        """Add a new face to the database"""
        if face_box is None:
            faces = self.detector.detect_faces(image)
            if not faces:
                logger.warning("No face detected in the image")
                return False
            face_box = faces[0]['box']  # Use the first detected face
        
        x, y, w, h = face_box
        face_img = image[y:y+h, x:x+w]
        
        if face_img.size == 0:
            logger.warning("Invalid face region")
            return False
        
        features = self.recognizer.extract_features(face_img)
        
        if name not in self.known_faces:
            self.known_faces[name] = []
        
        self.known_faces[name].append(features)
        self.save_database()
        logger.info(f"Added face for {name}")
        return True
    
    def recognize_face(self, face_img, threshold=0.6):
        """Recognize a face by comparing with known faces"""
        features = self.recognizer.extract_features(face_img)
        
        best_match = None
        best_distance = float('inf')
        
        for name, face_features_list in self.known_faces.items():
            for known_features in face_features_list:
                # Calculate cosine similarity
                similarity = np.dot(features, known_features) / (
                    np.linalg.norm(features) * np.linalg.norm(known_features)
                )
                distance = 1 - similarity
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
        
        if best_distance < threshold:
            return best_match, best_distance
        else:
            return "Unknown", best_distance
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        faces = self.detector.detect_faces(frame)
        results = []
        
        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size > 0:
                name, distance = self.recognize_face(face_img)
                results.append({
                    'box': face['box'],
                    'name': name,
                    'confidence': face['confidence'],
                    'distance': distance
                })
        
        return results
    
    def draw_results(self, frame, results):
        """Draw detection and recognition results on frame"""
        for result in results:
            x, y, w, h = result['box']
            name = result['name']
            confidence = result['confidence']
            
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw name and confidence
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description='Face Detection and Recognition for Raspberry Pi 5')
    parser.add_argument('--mode', choices=['camera', 'image', 'add_face'], default='camera',
                       help='Operation mode')
    parser.add_argument('--input', help='Input image path (for image mode)')
    parser.add_argument('--name', help='Name for adding face')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Initialize face recognizer
    face_recognizer = FaceRecognizer()
    
    if args.mode == 'camera':
        # Camera mode
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for RPi
        
        logger.info("Starting camera mode. Press 'q' to quit, 'a' to add current face")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Process frame
            results = face_recognizer.process_frame(frame)
            
            # Draw results
            frame = face_recognizer.draw_results(frame, results)
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # Add face mode
                name = input("Enter name for this face: ")
                if name and results:
                    face_recognizer.add_face(frame, name, results[0]['box'])
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.mode == 'image':
        # Image mode
        if not args.input:
            logger.error("Please provide input image path with --input")
            return
        
        image = cv2.imread(args.input)
        if image is None:
            logger.error("Could not load image")
            return
        
        results = face_recognizer.process_frame(image)
        image = face_recognizer.draw_results(image, results)
        
        cv2.imshow('Face Recognition', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif args.mode == 'add_face':
        # Add face mode
        if not args.input or not args.name:
            logger.error("Please provide both --input and --name for adding faces")
            return
        
        image = cv2.imread(args.input)
        if image is None:
            logger.error("Could not load image")
            return
        
        success = face_recognizer.add_face(image, args.name)
        if success:
            print(f"Successfully added face for {args.name}")
        else:
            print("Failed to add face")

if __name__ == "__main__":
    main()
