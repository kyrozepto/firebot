import cv2
import torch
import numpy as np
import time
import yaml  # Import YAML for config file
import asyncio  # Import asyncio for asynchronous operations
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

# --- Configuration Loading ---
def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

# --- Asynchronous Firebase Update ---
async def send_command_to_firebase_async(command_code, command_path):
    """Sends the command code to Firebase asynchronously."""
    try:
        loop = asyncio.get_event_loop()
        ref = db.reference(command_path)
        # Use asyncio.to_thread to run the blocking I/O operation in a separate thread
        await loop.run_in_executor(None, ref.set, command_code)
    except Exception as e:
        print(f"Error sending command to Firebase: {e}")

# --- Firebase Initialization ---
def initialize_firebase(cred_path, db_url):
    """Initializes Firebase Admin SDK."""
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {'databaseURL': db_url})
        print("Firebase initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

# --- Model Loading ---
def load_yolo_model(model_path):
    """Loads the YOLOv8 model."""
    try:
        model = YOLO(model_path)
        print(f"YOLO model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def initialize_depth_model(model_name):
    """Initializes the DepthAnythingV2 model and processor."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
        print(f"DepthAnythingV2 model loaded successfully on {device}.")
        return model, processor, device
    except Exception as e:
        print(f"Error initializing depth model: {e}")
        return None, None, None

# --- Core Logic ---
def get_depth_for_box(depth_map, box):
    """Calculates the median depth for a given bounding box."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_map.shape[1] - 1, x2), min(depth_map.shape[0] - 1, y2)

    if x1 >= x2 or y1 >= y2:
        return float('inf')

    depth_roi = depth_map[y1:y2, x1:x2]
    return np.median(depth_roi) if depth_roi.size > 0 else float('inf')

def get_robot_instruction(detections, depth_map, frame_width, config):
    """Determines the robot instruction based on detections and depth."""
    detected_fire_objects = []
    detected_smoke_objects = []

    if detections and detections[0].boxes is not None:
        for r in detections:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < config['yolo']['model_confidence']:
                    continue

                cls = int(box.cls[0])
                obj_box = box.xyxy[0]
                obj_center_x = (obj_box[0] + obj_box[2]) / 2
                obj_depth = get_depth_for_box(depth_map, obj_box)
                
                obj_data = {'center_x': obj_center_x, 'confidence': conf, 'box': obj_box, 'depth': obj_depth}

                if cls == config['class_ids']['fire']:
                    detected_fire_objects.append(obj_data)
                elif cls == config['class_ids']['smoke']:
                    detected_smoke_objects.append(obj_data)

    target_objects = detected_fire_objects if detected_fire_objects else detected_smoke_objects
    threat_type = "Fire" if detected_fire_objects else "Smoke"

    if not target_objects:
        return config['commands']['stop'], "No Threat (Stop)", None, None

    target = max(target_objects, key=lambda x: x['confidence'])
    print(f"{threat_type} detected at x_center: {target['center_x']:.2f}, Depth: {target['depth']:.2f}")

    if target['depth'] < config['depth_model']['safe_distance_threshold']:
        return config['commands']['stop'], f"{threat_type} Too Close! Stop!", target['box'], target['depth']
    
    if target['center_x'] < config['frame_division']['left_end'] * frame_width:
        return config['commands']['left'], f"{threat_type} Left", target['box'], target['depth']
    elif target['center_x'] > config['frame_division']['right_start'] * frame_width:
        return config['commands']['right'], f"{threat_type} Right", target['box'], target['depth']
    else:
        return config['commands']['forward'], f"{threat_type} Center", target['box'], target['depth']

# --- Main Asynchronous Execution ---
async def main():
    config = load_config()
    if config is None:
        exit("Configuration failed to load. Exiting.")

    if not initialize_firebase(config['firebase']['cred_path'], config['firebase']['db_url']):
        exit("Firebase initialization failed. Exiting.")

    yolo_model = load_yolo_model(config['yolo']['model_path'])
    if yolo_model is None:
        exit("YOLO model loading failed. Exiting.")

    depth_model, depth_processor, device = initialize_depth_model(config['depth_model']['name'])
    if depth_model is None:
        exit("Depth model loading failed. Exiting.")

    cap = cv2.VideoCapture(config['video_source'])
    if not cap.isOpened():
        print(f"Error: Could not open video source: {config['video_source']}")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source opened. Resolution: {frame_width}x{frame_height}")

    loop = asyncio.get_event_loop()
    last_command_sent_time = time.time()
    current_instruction_code = config['commands']['stop']
    await send_command_to_firebase_async(current_instruction_code, config['firebase']['command_path'])

    try:
        while True:
            # Run blocking frame capture in a separate thread
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                print("End of video stream or error reading frame.")
                break

            # --- Depth Estimation ---
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = depth_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = depth_model(**inputs)
                prediction = torch.nn.functional.interpolate(
                    outputs.predicted_depth.unsqueeze(1),
                    size=(frame_height, frame_width),
                    mode="bicubic",
                    align_corners=False,
                )
            depth_map = prediction.squeeze().cpu().numpy()

            # --- Object Detection ---
            yolo_results = yolo_model(frame, verbose=False, conf=config['yolo']['model_confidence'])

            # --- Decision Making ---
            instruction_code, instruction_label, target_box, target_depth = get_robot_instruction(yolo_results, depth_map, frame_width, config)

            # --- Asynchronous Firebase Command ---
            current_time = time.time()
            if instruction_code != current_instruction_code or (current_time - last_command_sent_time > config['command_interval']):
                asyncio.create_task(send_command_to_firebase_async(instruction_code, config['firebase']['command_path']))
                current_instruction_code = instruction_code
                last_command_sent_time = current_time
                print(f"Sending command: {instruction_label} ({instruction_code})")
            
            # --- Visualization ---
            overlay = yolo_results[0].plot() if yolo_results else frame.copy()
            if target_box is not None:
                x1, y1, x2, y2 = map(int, target_box)
                box_color = (0, 0, 255) if "Stop" in instruction_label else (0, 255, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
                label_text = f"{instruction_label.split('(')[0].strip()} D:{target_depth:.2f}"
                cv2.putText(overlay, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            cv2.putText(overlay, f"Command: {instruction_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.line(overlay, (int(frame_width * config['frame_division']['left_end']), 0), (int(frame_width * config['frame_division']['left_end']), frame_height), (0,0,255),1)
            cv2.line(overlay, (int(frame_width * config['frame_division']['right_start']), 0), (int(frame_width * config['frame_division']['right_start']), frame_height), (0,0,255),1)
            
            depth_vis = cv2.normalize(depth_map, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            
            cv2.imshow("Firefighter Robot Surveillance", overlay)
            cv2.imshow("Depth Map", depth_vis_colored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    finally:
        await send_command_to_firebase_async(config['commands']['stop'], config['firebase']['command_path'])
        print("Sent final STOP command.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())