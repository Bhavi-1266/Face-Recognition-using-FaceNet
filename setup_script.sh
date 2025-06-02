#!/bin/bash
# Setup script for Face Detection and Recognition on Raspberry Pi 5

echo "Setting up Face Detection and Recognition for Raspberry Pi 5..."

# Create project directory
PROJECT_DIR="face_recognition_rpi"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-opencv python3-pip python3-dev
sudo apt install -y libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y libatlas-base-dev libjasper-dev

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install Pillow
pip install argparse

# Create directories
mkdir -p models
mkdir -p test_images

# Download SSD MobileNet face detection model
echo "Downloading face detection models..."
cd models

# Download SSD face detection model files
wget -O deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget -O res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

echo "Models downloaded successfully!"

cd ..

# Create requirements.txt
cat > requirements.txt << EOL
opencv-python==4.8.1.78
numpy==1.24.3
Pillow>=8.0.0
EOL

# Create a simple test script
cat > test_camera.py << 'EOL'
#!/usr/bin/env python3
import cv2
import sys

def test_camera():
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    print("Camera opened successfully!")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    test_camera()
EOL

chmod +x test_camera.py

# Create usage instructions
cat > README.md << 'EOL'
# Face Detection and Recognition for Raspberry Pi 5

## Setup Complete!

### Files Created:
- `face_recognition.py` - Main face detection and recognition program
- `test_camera.py` - Camera test script
- `models/` - Directory containing detection models
- `requirements.txt` - Python dependencies

### Usage:

1. **Test Camera First:**
   ```bash
   python3 test_camera.py
   ```

2. **Run Face Detection (Camera Mode):**
   ```bash
   python3 face_recognition.py --mode camera
   ```

3. **Add New Faces:**
   ```bash
   python3 face_recognition.py --mode add_face --input photo.jpg --name "John Doe"
   ```

4. **Process Single Image:**
   ```bash
   python3 face_recognition.py --mode image --input photo.jpg
   ```

### Controls in Camera Mode:
- Press 'q' to quit
- Press 'a' to add the currently detected face to database

### Performance Optimization for RPi 5:
- Lower camera resolution (640x480) for better performance
- Reduced FPS (15) to prevent overload
- Optimized detection confidence threshold
- Lightweight feature extraction fallback

### Troubleshooting:
1. If camera doesn't work, try different camera indices (0, 1, 2...)
2. For USB cameras, ensure proper power supply
3. If detection is slow, reduce image resolution further
4. Check that all models are downloaded in the models/ directory

### MobileFaceNet Model (Optional Enhancement):
To use the full MobileFaceNet model for better recognition:
1. Download MobileFaceNet ONNX model
2. Place it in models/mobilefacenet.onnx
3. The system will automatically use it instead of basic features

EOL

echo ""
echo "Setup complete! Your face detection and recognition system is ready."
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Test camera: python3 test_camera.py"
echo "3. Run face recognition: python3 face_recognition.py --mode camera"
echo ""
echo "Check README.md for detailed usage instructions."
