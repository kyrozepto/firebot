# FireBot Navigation

A computer vision fire and smoke detection system that controls a robot to navigate towards detected threats. The system uses fine-tuned YOLOv8n and Firebase Realtime Database.

### Features

- Real-time fire and smoke detection
- Autonomous robot navigation towards detected threats
- Firebase integration for robot control
- Visual feedback with bounding boxes

### Prerequisites

- Python 3.8 or higher
- OpenCV
- Ultralytics YOLOv8
- Firebase Admin SDK
- A webcam or video source
- Firebase project with Realtime Database

## Installation

1. Clone this repository
   ```
   git clone https://github.com/kyrozepto/firebot.git
   cd firebot
   ```

2. Create a virtual environment
   ```
   python -m venv myenv
   ```

3. Activate the virtual environment

   - On Windows:
     ```
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source myenv/bin/activate
     ```

4. Install required packages
   ```
   pip install opencv-python ultralytics firebase-admin
   ```

## Configuration

1. **Firebase Setup**:
   - Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
   - Set up a Realtime Database
   - Generate a service account key (Project Settings > Service Accounts > Generate new private key)
   - Save the key as `serviceAccountKey.json` in the project root

2. **YOLOv8 Model**:
   - Download or train your own YOLOv8 model for fire and smoke detection
   - Place the model file (default: `yolov8n-200e-v0.2.pt`) in the project root

3. **Update Configuration**:
   - Open `navigate.py` and modify the following variables if needed:
     ```python
     MODEL_PATH = 'yolov8n-200e-v0.2.pt'
     FIREBASE_CRED_PATH = 'serviceAccountKey.json'
     FIREBASE_DB_URL = 'https://project-id.firebasedatabase.app'
     FIREBASE_COMMAND_PATH = '/firebot/command'
     ```

## Run the Project

1. Ensure your webcam is connected or specify a video file path in the code
2. Run the navigation system:
   ```
   python navigate.py
   ```
3. The system will:
   - Display a video feed with detection results
   - Send navigation commands to Firebase
   - Show visual guides for navigation zones

4. Press 'q' to exit the program

## Customization

You can adjust the following parameters in `navigate.py`:
- `CONFIDENCE_THRESHOLD`: Minimum confidence for valid detections
- `FRAME_SECTION_LEFT_END` and `FRAME_SECTION_RIGHT_START`: Frame division for navigation decisions

### FireBot Command Codes

- `0`: Stop
- `1`: Move Forward
- `2`: Turn Left
- `3`: Turn Right

### Firebase Database Structure

The system expects the following database structure:
```
/firebot
    /command: <integer>  # Command code (0-3)
```
