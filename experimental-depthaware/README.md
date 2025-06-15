# Robotic Firebot - Experimental Depth-Aware Version

This project implements a robotic firebot system with depth-aware capabilities using YOLOv8 and various computer vision techniques.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for GPU acceleration)
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd robotic-firebot/experimental-depthaware
```

### 2. Environment Setup

#### Option A: Using Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Conda Environment

```bash
# Create a conda environment
conda create -n firebot python=3.8
conda activate firebot

# Install dependencies
pip install -r requirements.txt
```

### 3. GPU Setup (Optional)

If you have a CUDA-capable GPU and want to use GPU acceleration:

1. Install CUDA Toolkit (compatible with your GPU)
2. Install cuDNN
3. Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

To verify GPU availability, run:
```bash
python check_gpu.py
```

### 4. Configuration

1. Ensure all required model files are present:
   - yolov8n-firebot.pt
   - yolov8n.pt
   - yolov8n-seg.pt
   - yolo11n-seg.pt
   - yolov8n-200e-v0.2.pt

2. Configure your settings in `config.yaml`

3. Set up Firebase credentials:
   - Place your `ServiceAccountKey.json` in the project root directory

## Running the Project

### Using the Batch File (Windows)

Simply run:
```bash
run_robot.bat
```

### Manual Execution

```bash
python navigate2depth.py
```

## Project Structure

- `navigate2depth.py`: Main application file
- `config.yaml`: Configuration settings
- `check_gpu.py`: GPU verification utility
- `requirements.txt`: Project dependencies
- Model files:
  - `yolov8n-firebot.pt`: Custom trained model
  - `yolov8n.pt`: Base YOLOv8 model
  - `yolov8n-seg.pt`: Segmentation model
  - `yolo11n-seg.pt`: Alternative segmentation model
  - `yolov8n-200e-v0.2.pt`: Fine tuned model variant for fire and smoke object
 
## Related Paper
Vision Transformers for Dense Prediction
https://arxiv.org/abs/2103.13413
