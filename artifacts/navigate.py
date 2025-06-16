import cv2
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
import time
import numpy as np # For drawing bounding boxes

# --- Configuration ---
MODEL_PATH = 'yolov8n-200e-v0.2.pt'  # Path to your trained YOLOv8n model
FIREBASE_CRED_PATH = 'serviceAccountKey.json' # Path to your Firebase service account key
FIREBASE_DB_URL = 'https://firebot-sfy832-default-rtdb.asia-southeast1.firebasedatabase.app'  # Your Firebase Realtime Database URL
FIREBASE_COMMAND_PATH = '/firebot/command' # Path in Firebase to send commands

# Class IDs from your model (adjust if different)
CLASS_ID_FIRE = 80
CLASS_ID_SMOKE = 81

# Instruction codes
CMD_STOP = 0
CMD_FORWARD = 1
CMD_LEFT = 2
CMD_RIGHT = 3

# Frame division for decision making (thirds)
FRAME_SECTION_LEFT_END = 1/3
FRAME_SECTION_RIGHT_START = 2/3

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# --- Firebase Initialization ---
def initialize_firebase():
    """Initializes Firebase Admin SDK."""
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred, {
            'databaseURL': FIREBASE_DB_URL
        })
        print("Firebase initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

# --- YOLO Model Loading ---
def load_yolo_model(model_path):
    """Loads the YOLOv8 model."""
    try:
        model = YOLO(model_path)
        print(f"YOLO model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# --- Decision Logic ---
def get_robot_instruction(detections, frame_width):
    """
    Determines the robot instruction based on detections.
    Prioritizes fire, then smoke.
    """
    detected_fire_objects = []
    detected_smoke_objects = []

    if detections and detections[0].boxes is not None: # Check if detections[0].boxes is not None
        for r in detections:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                # Calculate center x of the bounding box
                obj_center_x = (x1 + x2) / 2
                obj_data = {'center_x': obj_center_x, 'confidence': conf, 'box': (x1, y1, x2, y2)}

                if cls == CLASS_ID_FIRE:
                    detected_fire_objects.append(obj_data)
                elif cls == CLASS_ID_SMOKE:
                    detected_smoke_objects.append(obj_data)

    # Prioritize fire
    if detected_fire_objects:
        # For simplicity, consider the fire object with the highest confidence or largest area
        # Here, let's take the first detected fire object (can be improved)
        target_fire = max(detected_fire_objects, key=lambda x: x['confidence']) # Or use area: (x2-x1)*(y2-y1)
        
        print(f"Fire detected at x_center: {target_fire['center_x']:.2f}")
        if target_fire['center_x'] < frame_width * FRAME_SECTION_LEFT_END:
            return CMD_LEFT, "Fire Left", target_fire['box']
        elif target_fire['center_x'] > frame_width * FRAME_SECTION_RIGHT_START:
            return CMD_RIGHT, "Fire Right", target_fire['box']
        else:
            return CMD_FORWARD, "Fire Center", target_fire['box'] # Go anyway, we will use the infra-red sensor to stop

    # If no fire, check for smoke
    elif detected_smoke_objects:
        target_smoke = max(detected_smoke_objects, key=lambda x: x['confidence'])
        print(f"Smoke detected at x_center: {target_smoke['center_x']:.2f}")
        if target_smoke['center_x'] < frame_width * FRAME_SECTION_LEFT_END:
            return CMD_LEFT, "Smoke Left", target_smoke['box']
        elif target_smoke['center_x'] > frame_width * FRAME_SECTION_RIGHT_START:
            return CMD_RIGHT, "Smoke Right", target_smoke['box']
        else:
            return CMD_FORWARD, "Smoke Center (Investigate)", target_smoke['box'] # Move forward to investigate smoke

    # No relevant detections
    return CMD_STOP, "No Threat (Stop)", None


# --- Firebase Update ---
def send_command_to_firebase(command_code):
    """Sends the command code to Firebase."""
    try:
        ref = db.reference(FIREBASE_COMMAND_PATH)
        ref.set(command_code)
        # print(f"Sent command to Firebase: {command_code}")
    except Exception as e:
        print(f"Error sending command to Firebase: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    if not initialize_firebase():
        exit("Firebase initialization failed. Exiting.")

    model = load_yolo_model(MODEL_PATH)
    if model is None:
        exit("YOLO model loading failed. Exiting.")

    # --- Video Source Setup ---
    # Use 0 for webcam, or provide a video file path
    video_source = 0  # or "path/to/your/video.mp4"
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source opened. Resolution: {frame_width}x{frame_height}")

    last_command_sent_time = time.time()
    command_send_interval = 1.0  # Send command every 1 second to avoid flooding Firebase

    current_instruction_code = CMD_STOP # Initialize with STOP
    send_command_to_firebase(current_instruction_code) # Send initial STOP command

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            # Perform inference
            results = model(frame, verbose=False) # verbose=False to reduce console output

            # Get instruction based on detections
            instruction_code, instruction_label, target_box = get_robot_instruction(results, frame_width)

            # Send command to Firebase periodically
            current_time = time.time()
            if instruction_code != current_instruction_code or (current_time - last_command_sent_time > command_send_interval) :
                send_command_to_firebase(instruction_code)
                current_instruction_code = instruction_code
                last_command_sent_time = current_time
                print(f"Sending command: {instruction_label} ({instruction_code})")


            # --- Visualization (Optional) ---
            # Draw bounding box of the primary target
            if target_box:
                x1, y1, x2, y2 = target_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, instruction_label.split('(')[0].strip(), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display general instruction on frame
            cv2.putText(frame, f"Command: {instruction_label} ({instruction_code})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Display divisions
            cv2.line(frame, (int(frame_width * FRAME_SECTION_LEFT_END), 0), 
                     (int(frame_width * FRAME_SECTION_LEFT_END), frame_height), (0,0,255),1)
            cv2.line(frame, (int(frame_width * FRAME_SECTION_RIGHT_START), 0), 
                     (int(frame_width * FRAME_SECTION_RIGHT_START), frame_height), (0,0,255),1)


            cv2.imshow("Firefighter Robot Surveillance", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    finally:
        # Send a final STOP command before exiting
        send_command_to_firebase(CMD_STOP)
        print("Sent final STOP command.")
        cap.release()
        cv2.destroyAllWindows()