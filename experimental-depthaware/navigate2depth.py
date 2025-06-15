import cv2
import torch
import numpy as np
import time
import yaml
import asyncio
import math
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
        config['class_names'] = {
            str(config['class_ids']['fire']): 'Fire',
            str(config['class_ids']['smoke']): 'Smoke',
        }
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

# --- Device Selection ---
def get_device(config):
    """Determines whether to use GPU or CPU based on config and availability."""
    if config['gpu']['enabled'] and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# --- Asynchronous Firebase Update ---
async def send_command_to_firebase_async(command_code, command_path):
    """Sends the command code to Firebase asynchronously."""
    try:
        loop = asyncio.get_event_loop()
        ref = db.reference(command_path)
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

def initialize_depth_model(model_name, device):
    """Initializes the Depth model and processor."""
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
        print(f"Depth model loaded successfully on {device}.")
        return model, processor
    except Exception as e:
        print(f"Error initializing depth model: {e}")
        return None, None

# --- Core Logic ---
def get_median_depth_in_box(depth_map, box):
    """Calculates the median raw depth value for a given bounding box."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_map.shape[1] - 1, x2), min(depth_map.shape[0] - 1, y2)
    if x1 >= x2 or y1 >= y2:
        return 0
    depth_roi = depth_map[y1:y2, x1:x2]
    return np.median(depth_roi) if depth_roi.size > 0 else 0

# MODIFIED: This function now inverts the depth to get a real distance metric.
def process_frame_detections(yolo_results, depth_map, config):
    """Processes YOLO results to create a structured list of detected objects with depth."""
    detected_objects = []
    if not yolo_results or yolo_results[0].boxes is None:
        return detected_objects

    for r in yolo_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < config['yolo']['model_confidence']:
                continue

            cls = int(box.cls[0])
            obj_box = box.xyxy[0]
            
            # Get the raw inverse depth value from the model
            raw_depth_value = get_median_depth_in_box(depth_map, obj_box)
            
            # --- CRITICAL FIX: Invert the raw value to get a distance metric ---
            # We use a scaling factor to bring the distances into a nice range.
            # You can adjust the 25000 factor if needed.
            distance_metric = 0
            if raw_depth_value > 0:
                distance_metric = 25000 / raw_depth_value
            
            detected_objects.append({
                'class_id': cls,
                'class_name': config['class_names'].get(str(cls), f'Class_{cls}'),
                'center_x': (obj_box[0] + obj_box[2]) / 2,
                'confidence': conf,
                'box': obj_box,
                'raw_depth': raw_depth_value, # Keep for debugging if needed
                'distance_metric': distance_metric # Use this for all logic now
            })
            
    # CRITICAL FIX: Sort by the new distance_metric, ascending (closest first).
    detected_objects.sort(key=lambda x: x['distance_metric'])
    return detected_objects

# MODIFIED: This function now uses the intuitive distance_metric.
def get_robot_instruction_modified(detected_objects, frame_width, config):
    """
    Determines the robot's command based on a priority system using an intuitive distance metric.
    """
    fire_targets = [obj for obj in detected_objects if obj['class_id'] == config['class_ids']['fire']]
    obstacles = [obj for obj in detected_objects if obj['class_id'] not in [config['class_ids']['fire'], config['class_ids']['smoke']]]
    safe_dist_threshold = config['depth_model']['safe_distance_threshold']

    # --- Priority 1: Immediate Obstacle Avoidance ---
    # Logic is now more intuitive: is the distance LESS THAN the threshold?
    if obstacles and obstacles[0]['distance_metric'] < safe_dist_threshold:
        closest_obstacle = obstacles[0]
        if closest_obstacle['center_x'] < frame_width / 2:
            return config['commands']['right'], f"AVOID: Obstacle on Left ({closest_obstacle['class_name']})"
        else:
            return config['commands']['left'], f"AVOID: Obstacle on Right ({closest_obstacle['class_name']})"

    # --- Priority 2: Target (Fire) Navigation ---
    if fire_targets:
        main_fire_target = fire_targets[0]
        if main_fire_target['distance_metric'] < safe_dist_threshold:
             return config['commands']['stop'], "STOP: Fire too close!"
        if main_fire_target['center_x'] < frame_width * config['frame_division']['left_end']:
            return config['commands']['left'], "TARGET: Fire on Left"
        elif main_fire_target['center_x'] > frame_width * config['frame_division']['right_start']:
            return config['commands']['right'], "TARGET: Fire on Right"
        else:
            return config['commands']['forward'], "TARGET: Fire Ahead"

    # --- Priority 3: Cautious Exploration ---
    center_obstacles = [obs for obs in obstacles if (frame_width * config['frame_division']['left_end'] < obs['center_x'] < frame_width * config['frame_division']['right_start'])]
    if not center_obstacles:
        return config['commands']['forward'], "EXPLORE: Path clear"

    # --- Priority 4: Default to Stop ---
    return config['commands']['stop'], "STOP: Path blocked or no target"

# --- Visualization ---
# MODIFIED: This function now uses the intuitive distance_metric.
def get_world_coords(obj, frame_width, fov_degrees):
    """
    Projects a detected object from image space to a top-down world coordinate system.
    """
    # Use the new distance_metric for forward distance
    forward_distance = obj['distance_metric']
    if forward_distance <= 0:
        return None
        
    camera_center_x = frame_width / 2
    pixel_offset = obj['center_x'] - camera_center_x
    fov_radians = math.radians(fov_degrees)
    focal_length = (frame_width / 2) / math.tan(fov_radians / 2)
    lateral_distance = (pixel_offset * forward_distance) / focal_length
    return (lateral_distance, forward_distance)

# MODIFIED: This function now correctly scales the map.
def create_top_down_map(detected_objects, frame_width, config):
    """Creates a 2D top-down map visualization of detected objects."""
    map_size = (500, 500)
    map_image = np.zeros((map_size[1], map_size[0], 3), dtype=np.uint8)
    map_config = config['map_view']
    map_scale = map_config['scale']
    
    robot_x, robot_y = map_size[0] // 2, map_size[1] - 30
    pts = np.array([[robot_x, robot_y - 10], [robot_x - 10, robot_y + 10], [robot_x + 10, robot_y + 10]], np.int32)
    cv2.polylines(map_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(map_image, "Robot", (robot_x - 25, robot_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw grid lines based on the new distance metric
    for i in range(1, 10):
        dist_val = i * 20 # Represents 20, 40, 60... units of distance
        y_pos = robot_y - int(dist_val * map_scale)
        if y_pos < 0: break
        cv2.line(map_image, (0, y_pos), (map_size[0], y_pos), (50, 50, 50), 1)
        cv2.putText(map_image, f"{dist_val:.0f}", (5, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    for obj in detected_objects:
        world_coords = get_world_coords(obj, frame_width, map_config['camera_fov'])
        if world_coords is None: continue
        world_x, world_y = world_coords
        map_pixel_x = robot_x + int(world_x * map_scale)
        map_pixel_y = robot_y - int(world_y * map_scale)
        
        if 0 <= map_pixel_x < map_size[0] and 0 <= map_pixel_y < map_size[1]:
            color = (0, 0, 255) if obj['class_id'] == config['class_ids']['fire'] else (0, 255, 255)
            cv2.circle(map_image, (map_pixel_x, map_pixel_y), 5, color, -1)
            cv2.putText(map_image, obj['class_name'], (map_pixel_x + 7, map_pixel_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(map_image, "Top-Down Map", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return map_image

def visualize_depth(depth_map, frame_height, frame_width):
    """Visualizes the depth map with a colormap."""
    min_val, max_val = depth_map.min(), depth_map.max()
    if max_val > min_val:
      depth_vis = (depth_map - min_val) / (max_val - min_val) * 255.0
    else:
      depth_vis = np.zeros_like(depth_map)
    depth_vis = depth_vis.astype(np.uint8)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

# MODIFIED: This function now displays the intuitive distance metric.
def visualize_objects(frame, detected_objects, config):
    """Visualize detected objects with their bounding boxes and info."""
    overlay = frame.copy()
    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, obj['box'])
        box_color = (0, 0, 255) if obj['class_id'] == config['class_ids']['fire'] else (0, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
        
        # Display the new, intuitive distance metric
        info_text = f"{obj['class_name']}: {obj['distance_metric']:.1f}m"
        
        (w, h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1 - h - 10), (x1 + w, y1), box_color, -1)
        cv2.putText(overlay, info_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return overlay

# --- Main Asynchronous Execution ---
async def main():
    config = load_config()
    if config is None: exit("Configuration failed to load. Exiting.")

    device = get_device(config)

    if not initialize_firebase(config['firebase']['cred_path'], config['firebase']['db_url']):
        exit("Firebase initialization failed. Exiting.")

    yolo_model = load_yolo_model(config['yolo']['model_path'])
    if yolo_model is None: exit("YOLO model loading failed. Exiting.")

    depth_model, depth_processor = initialize_depth_model(config['depth_model']['name'], device)
    if depth_model is None: exit("Depth model loading failed. Exiting.")

    cap = cv2.VideoCapture(config['video_source'])
    if not cap.isOpened():
        print(f"Error: Could not open video source: {config['video_source']}")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source opened. Resolution: {frame_width}x{frame_height}")

    loop = asyncio.get_event_loop()
    last_command_sent_time = 0
    current_instruction_code = -1 
    await send_command_to_firebase_async(config['commands']['stop'], config['firebase']['command_path'])

    try:
        while True:
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                print("End of video stream or error reading frame.")
                break

            # --- Perception Stage ---
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = depth_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = depth_model(**inputs)
                prediction = torch.nn.functional.interpolate(
                    outputs.predicted_depth.unsqueeze(1),
                    size=(frame_height, frame_width),
                    mode="bicubic", align_corners=False,
                )
            depth_map = prediction.squeeze().cpu().numpy()
            yolo_results = yolo_model(frame, verbose=False, conf=config['yolo']['model_confidence'])
            
            # --- Centralized Processing ---
            detected_objects = process_frame_detections(yolo_results, depth_map, config)

            # --- Decision Stage ---
            instruction_code, instruction_label = get_robot_instruction_modified(
                detected_objects, frame_width, config)

            # --- Actuation Stage ---
            current_time = time.time()
            if instruction_code != current_instruction_code or (current_time - last_command_sent_time > config['command_interval']):
                asyncio.create_task(send_command_to_firebase_async(instruction_code, config['firebase']['command_path']))
                current_instruction_code = instruction_code
                last_command_sent_time = current_time
                print(f"[{time.strftime('%H:%M:%S')}] Command: {instruction_label} ({instruction_code})")
            
            # --- Visualization Stage ---
            overlay = visualize_objects(frame, detected_objects, config)
            cv2.putText(overlay, f"CMD: {instruction_label}", (10, frame_height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(overlay, (int(frame_width * config['frame_division']['left_end']), 0), 
                    (int(frame_width * config['frame_division']['left_end']), frame_height), (0,0,255), 1)
            cv2.line(overlay, (int(frame_width * config['frame_division']['right_start']), 0), 
                    (int(frame_width * config['frame_division']['right_start']), frame_height), (0,0,255), 1)
            
            depth_vis_colored = visualize_depth(depth_map, frame_height, frame_width)
            
            panels = [overlay, depth_vis_colored]
            
            if config['map_view']['enabled']:
                top_down_map = create_top_down_map(detected_objects, frame_width, config)
                map_resized = cv2.resize(top_down_map, (int(top_down_map.shape[1] * (frame_height / top_down_map.shape[0])), frame_height))
                panels.append(map_resized)
            
            combined_frame = cv2.hconcat(panels)
            cv2.imshow("Firefighter Robot - Navigation View", combined_frame)

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