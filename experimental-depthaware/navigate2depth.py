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
import matplotlib

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
    median_depth = np.median(depth_roi) if depth_roi.size > 0 else float('inf')
    
    # Convert depth to actual distance in centimeters
    # Assuming depth values are inversely proportional to distance
    # and using a scaling factor to get reasonable cm values
    if median_depth > 0:
        # Scale factor adjusted to get more accurate cm values
        distance_cm = (1000.0 / median_depth) * 100  # Multiply by 100 for more accurate cm values
        return min(distance_cm, 10000.0)  # Cap at 10000cm (100m) for stability
    return float('inf')

def get_robot_instruction(detections, depth_map, frame_width, config):
    """Enhanced decision making with obstacle avoidance and fire targeting."""
    detected_objects = []
    fire_detected = False
    fire_location = None
    obstacles = []

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
                
                obj_data = {
                    'class_id': cls,
                    'class_name': config['class_names'].get(str(cls), f'Class_{cls}'),
                    'center_x': obj_center_x,
                    'confidence': conf,
                    'box': obj_box,
                    'depth': obj_depth
                }
                detected_objects.append(obj_data)

                # Check if it's fire
                if cls == config['class_ids']['fire']:
                    fire_detected = True
                    fire_location = obj_data
                # Check if it's an obstacle (not fire or smoke)
                elif cls not in [config['class_ids']['fire'], config['class_ids']['smoke']]:
                    obstacles.append(obj_data)

    # Sort objects by depth (closest first)
    detected_objects.sort(key=lambda x: x['depth'], reverse=True)  # Reverse sort for actual distance
    obstacles.sort(key=lambda x: x['depth'], reverse=True)  # Reverse sort for actual distance

    # If fire is detected
    if fire_detected:
        # Check if there are obstacles between robot and fire
        if obstacles:
            closest_obstacle = obstacles[0]
            # If obstacle is too close and in the path to fire
            if closest_obstacle['depth'] < config['depth_model']['safe_distance_threshold']:
                # Determine which side has more space
                left_space = sum(1 for obj in obstacles if obj['center_x'] < frame_width * 0.4)
                right_space = sum(1 for obj in obstacles if obj['center_x'] > frame_width * 0.6)
                
                if left_space < right_space:
                    return config['commands']['right'], "Avoiding obstacle to reach fire", fire_location['box'], fire_location['depth']
                else:
                    return config['commands']['left'], "Avoiding obstacle to reach fire", fire_location['box'], fire_location['depth']
        
        # No obstacles or they're far enough, move towards fire
        if fire_location['center_x'] < frame_width * config['frame_division']['left_end']:
            return config['commands']['left'], "Fire Left", fire_location['box'], fire_location['depth']
        elif fire_location['center_x'] > frame_width * config['frame_division']['right_start']:
            return config['commands']['right'], "Fire Right", fire_location['box'], fire_location['depth']
        else:
            return config['commands']['forward'], "Fire Center", fire_location['box'], fire_location['depth']

    # No fire detected, handle obstacles
    if obstacles:
        closest_obstacle = obstacles[0]
        if closest_obstacle['depth'] < config['depth_model']['safe_distance_threshold']:
            if closest_obstacle['center_x'] < frame_width * 0.4:
                return config['commands']['right'], f"Avoiding {closest_obstacle['class_name']}", closest_obstacle['box'], closest_obstacle['depth']
            else:
                return config['commands']['left'], f"Avoiding {closest_obstacle['class_name']}", closest_obstacle['box'], closest_obstacle['depth']

    return config['commands']['stop'], "No immediate threats", None, None

def visualize_depth(depth_map, frame_height, frame_width):
    """Enhanced depth visualization with min/max points and values."""
    # Invert depth map for visualization (darker = closer)
    depth_map_inv = 1.0 / (depth_map + 1e-6)  # Add small epsilon to avoid division by zero
    min_val = depth_map_inv.min()
    max_val = depth_map_inv.max()
    
    min_loc = np.unravel_index(np.argmin(depth_map_inv, axis=None), depth_map_inv.shape)
    max_loc = np.unravel_index(np.argmax(depth_map_inv, axis=None), depth_map_inv.shape)
    
    # Convert to actual distances in cm with adjusted scaling
    min_dist_cm = (1000.0 / (depth_map[min_loc] + 1e-6)) * 100
    max_dist_cm = (1000.0 / (depth_map[max_loc] + 1e-6)) * 100
    
    # Normalize depth map for visualization
    depth_vis = (depth_map_inv - min_val) / (max_val - min_val) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    
    # Apply colormap
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_vis_colored = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    # Add min/max point markers and values
    cv2.putText(depth_vis_colored, f'Closest: {min_dist_cm:.1f}cm', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(depth_vis_colored, f'Farthest: {max_dist_cm:.1f}cm', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw circles at min/max points
    cv2.circle(depth_vis_colored, (min_loc[1], min_loc[0]), 5, (0, 255, 0), -1)
    cv2.circle(depth_vis_colored, (max_loc[1], max_loc[0]), 5, (255, 0, 255), -1)
    
    return depth_vis_colored

def visualize_objects(frame, detected_objects, config):
    """Visualize detected objects with their information."""
    overlay = frame.copy()
    y_offset = 30
    line_height = 25

    for obj in detected_objects:
        # Draw bounding box
        x1, y1, x2, y2 = map(int, obj['box'])
        box_color = (0, 0, 255) if obj['class_id'] == config['class_ids']['fire'] else (0, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

        # Prepare object info text
        info_text = f"{obj['class_name']}: {obj['confidence']:.2f}, {obj['depth']:.1f}cm"
        
        # Draw text background
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlay, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), box_color, -1)
        
        # Draw text
        cv2.putText(overlay, info_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add to info panel
        cv2.putText(overlay, info_text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        y_offset += line_height

    return overlay

# --- Main Asynchronous Execution ---
async def main():
    config = load_config()
    if config is None:
        exit("Configuration failed to load. Exiting.")

    # Add class names to config if not present
    if 'class_names' not in config:
        config['class_names'] = {
            str(config['class_ids']['fire']): 'Fire',
            str(config['class_ids']['smoke']): 'Smoke',
            # Add more class names as needed
        }

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

    # Add margin for visualization
    margin_width = 50
    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255

    loop = asyncio.get_event_loop()
    last_command_sent_time = time.time()
    current_instruction_code = config['commands']['stop']
    await send_command_to_firebase_async(current_instruction_code, config['firebase']['command_path'])

    try:
        while True:
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
            instruction_code, instruction_label, target_box, target_depth = get_robot_instruction(
                yolo_results, depth_map, frame_width, config)

            # --- Asynchronous Firebase Command ---
            current_time = time.time()
            if instruction_code != current_instruction_code or (current_time - last_command_sent_time > config['command_interval']):
                asyncio.create_task(send_command_to_firebase_async(instruction_code, config['firebase']['command_path']))
                current_instruction_code = instruction_code
                last_command_sent_time = current_time
                print(f"Sending command: {instruction_label} ({instruction_code})")
            
            # --- Enhanced Visualization ---
            # Get all detected objects with their information
            detected_objects = []
            if yolo_results and yolo_results[0].boxes is not None:
                for r in yolo_results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf < config['yolo']['model_confidence']:
                            continue
                        cls = int(box.cls[0])
                        obj_box = box.xyxy[0]
                        obj_depth = get_depth_for_box(depth_map, obj_box)
                        detected_objects.append({
                            'class_id': cls,
                            'class_name': config['class_names'].get(str(cls), f'Class_{cls}'),
                            'confidence': conf,
                            'box': obj_box,
                            'depth': obj_depth
                        })

            # Visualize objects with their information
            overlay = visualize_objects(frame, detected_objects, config)

            # Add command and navigation lines
            cv2.putText(overlay, f"Command: {instruction_label}", (10, frame_height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.line(overlay, (int(frame_width * config['frame_division']['left_end']), 0), 
                    (int(frame_width * config['frame_division']['left_end']), frame_height), (0,0,255),1)
            cv2.line(overlay, (int(frame_width * config['frame_division']['right_start']), 0), 
                    (int(frame_width * config['frame_division']['right_start']), frame_height), (0,0,255),1)
            
            # Enhanced depth visualization
            depth_vis_colored = visualize_depth(depth_map, frame_height, frame_width)
            
            # Combine frames with margin
            combined_frame = cv2.hconcat([overlay, split_region, depth_vis_colored])
            cv2.imshow("Firefighter Robot Surveillance", combined_frame)

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