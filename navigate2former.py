import cv2
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
import time
import numpy as np
from PIL import Image # For Mask2Former
import torch # For Mask2Former

# --- Mask2Former Imports ---
try:
    from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
    MASK2FORMER_AVAILABLE = True
except ImportError:
    print("Transformers library not found. Mask2Former functionality will be disabled.")
    print("Install it with: pip install transformers torch")
    MASK2FORMER_AVAILABLE = False

# --- Configuration ---
YOLO_MODEL_PATH = 'yolov8n-200e-v0.2.pt' # Path to your trained YOLOv8n model
MASK2FORMER_MODEL_NAME = "facebook/mask2former-swin-tiny-coco-instance" # Or another suitable model

FIREBASE_CRED_PATH = 'serviceAccountKey.json' # Path to your Firebase service account key
FIREBASE_DB_URL = 'https://firebot-sfy832-default-rtdb.asia-southeast1.firebasedatabase.app'  # Your Firebase Realtime Database URL
FIREBASE_COMMAND_PATH = '/firebot/command' # Path in Firebase to send commands

# Class IDs from your YOLO model (adjust if different)
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

# Confidence thresholds
YOLO_CONFIDENCE_THRESHOLD = 0.5
PERSON_CONFIDENCE_THRESHOLD = 0.7 # Confidence for person detection by Mask2Former

# Device for PyTorch models (Mask2Former)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} for Mask2Former")

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

# --- Mask2Former Model Loading ---
def load_mask2former_model(model_name):
    """Loads the Mask2Former model and processor."""
    if not MASK2FORMER_AVAILABLE:
        return None, None, -1

    try:
        processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(DEVICE)
        model.eval() # Set model to evaluation mode

        # Find the "person" class ID for COCO instance segmentation
        # This can vary, so it's good to check `model.config.id2label`
        person_class_id = -1
        for k, v in model.config.id2label.items():
            if v.lower() == 'person':
                person_class_id = k
                break
        if person_class_id == -1:
            print(f"Warning: 'person' class not found in Mask2Former model config ({model_name}). Person segmentation might not work.")
            print("Available labels:", model.config.id2label)
        else:
            print(f"Mask2Former model '{model_name}' loaded. Person class ID: {person_class_id}")
        return processor, model, person_class_id
    except Exception as e:
        print(f"Error loading Mask2Former model: {e}")
        return None, None, -1


# --- Decision Logic (YOLO-based) ---
def get_robot_instruction(yolo_detections, frame_width):
    """
    Determines the robot instruction based on YOLO fire/smoke detections.
    Prioritizes fire, then smoke.
    """
    detected_fire_objects = []
    detected_smoke_objects = []

    if yolo_detections and yolo_detections[0].boxes is not None:
        for r in yolo_detections:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if conf < YOLO_CONFIDENCE_THRESHOLD:
                    continue

                obj_center_x = (x1 + x2) / 2
                obj_data = {'center_x': obj_center_x, 'confidence': conf, 'box': (x1, y1, x2, y2)}

                if cls == CLASS_ID_FIRE:
                    detected_fire_objects.append(obj_data)
                elif cls == CLASS_ID_SMOKE:
                    detected_smoke_objects.append(obj_data)

    if detected_fire_objects:
        target_fire = max(detected_fire_objects, key=lambda x: x['confidence'])
        # print(f"Fire detected at x_center: {target_fire['center_x']:.2f}")
        if target_fire['center_x'] < frame_width * FRAME_SECTION_LEFT_END:
            return CMD_LEFT, "Fire Left", target_fire['box']
        elif target_fire['center_x'] > frame_width * FRAME_SECTION_RIGHT_START:
            return CMD_RIGHT, "Fire Right", target_fire['box']
        else:
            return CMD_FORWARD, "Fire Center", target_fire['box']

    elif detected_smoke_objects:
        target_smoke = max(detected_smoke_objects, key=lambda x: x['confidence'])
        # print(f"Smoke detected at x_center: {target_smoke['center_x']:.2f}")
        if target_smoke['center_x'] < frame_width * FRAME_SECTION_LEFT_END:
            return CMD_LEFT, "Smoke Left", target_smoke['box']
        elif target_smoke['center_x'] > frame_width * FRAME_SECTION_RIGHT_START:
            return CMD_RIGHT, "Smoke Right", target_smoke['box']
        else:
            return CMD_FORWARD, "Smoke Center (Investigate)", target_smoke['box']

    return CMD_STOP, "No Threat (Stop)", None


# --- Mask2Former Person Segmentation ---
# --- Mask2Former Person Segmentation ---
def get_person_masks(frame_bgr, m2f_processor, m2f_model, target_class_id):
    """
    Performs person segmentation using Mask2Former.
    Returns a list of binary masks for detected persons.
    """
    if m2f_processor is None or m2f_model is None or target_class_id == -1:
        return []

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    original_size = pil_image.size[::-1] # (height, width)

    inputs = m2f_processor(images=pil_image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = m2f_model(**inputs)

    # target_sizes must be a list of tuples (height, width)
    results = m2f_processor.post_process_instance_segmentation(
        outputs, target_sizes=[original_size], threshold=PERSON_CONFIDENCE_THRESHOLD
    )[0] # We process one image at a time

    person_masks_list = []
    if "segmentation" in results and "segments_info" in results:
        # For instance segmentation with transformers processor,
        # `results["segmentation"]` is typically a 2D tensor (H, W) where each pixel
        # value is an instance ID.
        instance_id_map_tensor = results["segmentation"] # This is (H, W) on DEVICE
        instance_id_map_np = instance_id_map_tensor.cpu().numpy().astype(np.int32)
        segments_info = results["segments_info"]

        # print(f"Debug: instance_id_map_np shape: {instance_id_map_np.shape}, dtype: {instance_id_map_np.dtype}")
        # print(f"Debug: segments_info: {segments_info}")


        for seg_info in segments_info:
            # print(f"Debug: seg_info: {seg_info}") # To see what's detected
            if seg_info['label_id'] == target_class_id and seg_info['score'] >= PERSON_CONFIDENCE_THRESHOLD:
                instance_id = seg_info['id']
                # Create a binary mask for the current instance_id
                mask = (instance_id_map_np == instance_id) # This creates a boolean mask
                person_masks_list.append(mask.astype(np.uint8)) # Convert boolean to 0/1 uint8 mask
    # else:
        # print("Debug: 'segmentation' or 'segments_info' not in Mask2Former results or empty.")
        # if "segmentation" in results:
        #     print(f"Debug: results['segmentation'] shape: {results['segmentation'].shape}")
        # if "segments_info" in results:
        #     print(f"Debug: results['segments_info']: {results['segments_info']}")


    return person_masks_list


# --- Firebase Update ---
def send_command_to_firebase(command_code):
    """Sends the command code to Firebase."""
    try:
        ref = db.reference(FIREBASE_COMMAND_PATH)
        ref.set(command_code)
    except Exception as e:
        print(f"Error sending command to Firebase: {e}")

# --- Visualization Helpers ---
def draw_yolo_detections(frame, instruction_label, target_box):
    if target_box:
        x1, y1, x2, y2 = target_box
        color = (0, 255, 0) # Green for fire/smoke target
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, instruction_label.split('(')[0].strip(), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def draw_person_segmentations(frame, person_masks_list):
    if not person_masks_list:
        return

    # Create a color for person masks (e.g., blue)
    person_color = np.array([255, 0, 0], dtype=np.uint8) # Blue in BGR
    overlay = frame.copy()

    for mask_np in person_masks_list:
        # mask_np is (H, W) boolean or 0/1. Ensure it's boolean for indexing.
        bool_mask = mask_np.astype(bool)
        overlay[bool_mask] = person_color

    # Blend the overlay with the original frame
    alpha = 0.4 # Transparency of the mask
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Optionally, draw bounding boxes for persons too
    # for mask_np in person_masks_list:
    #     contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     for contour in contours:
    #         if cv2.contourArea(contour) > 200: # Filter small areas
    #             x, y, w, h = cv2.boundingRect(contour)
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #             cv2.putText(frame, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)


# --- Main Execution ---
if __name__ == "__main__":
    if not initialize_firebase():
        exit("Firebase initialization failed. Exiting.")

    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    if yolo_model is None:
        exit("YOLO model loading failed. Exiting.")

    m2f_processor, m2f_model, m2f_person_class_id = None, None, -1
    if MASK2FORMER_AVAILABLE:
        m2f_processor, m2f_model, m2f_person_class_id = load_mask2former_model(MASK2FORMER_MODEL_NAME)
        if m2f_model is None:
            print("Mask2Former failed to load. Person segmentation will be disabled.")

    video_source = 0
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source opened. Resolution: {frame_width}x{frame_height}")

    last_command_sent_time = time.time()
    command_send_interval = 1.0

    current_instruction_code = CMD_STOP
    send_command_to_firebase(current_instruction_code)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            # 1. YOLO Inference for Fire/Smoke
            yolo_results = yolo_model(frame, verbose=False)
            instruction_code, instruction_label, target_box = get_robot_instruction(yolo_results, frame_width)

            # 2. Mask2Former Inference for Person Segmentation
            person_masks = []
            if m2f_model and m2f_processor: # Only run if Mask2Former is loaded
                person_masks = get_person_masks(frame, m2f_processor, m2f_model, m2f_person_class_id)

            # 3. Send Command to Firebase (based on YOLO)
            current_time = time.time()
            if instruction_code != current_instruction_code or (current_time - last_command_sent_time > command_send_interval):
                send_command_to_firebase(instruction_code)
                current_instruction_code = instruction_code
                last_command_sent_time = current_time
                print(f"Sending command: {instruction_label} ({instruction_code})")
            # 4. Visualization
            # Draw YOLO detections (fire/smoke target)
            draw_yolo_detections(frame, instruction_label, target_box)

            # Draw Person segmentations from Mask2Former
            if person_masks:
                draw_person_segmentations(frame, person_masks)
                cv2.putText(frame, f"Persons: {len(person_masks)}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


            # Display general instruction on frame
            cv2.putText(frame, f"Command: {instruction_label} ({instruction_code})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display frame sections for robot control
            cv2.line(frame, (int(frame_width * FRAME_SECTION_LEFT_END), 0),
                     (int(frame_width * FRAME_SECTION_LEFT_END), frame_height), (200,200,200),1)
            cv2.line(frame, (int(frame_width * FRAME_SECTION_RIGHT_START), 0),
                     (int(frame_width * FRAME_SECTION_RIGHT_START), frame_height), (200,200,200),1)

            cv2.imshow("Firefighter Robot Surveillance", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    finally:
        send_command_to_firebase(CMD_STOP)
        print("Sent final STOP command.")
        cap.release()
        cv2.destroyAllWindows()
        if MASK2FORMER_AVAILABLE and m2f_model: # Clean up PyTorch model if loaded
            del m2f_model
            del m2f_processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()