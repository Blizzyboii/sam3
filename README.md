# SAM3 Setup Guide for macOS

I'm writing this guide cause I spent too much time setting up the new SAM3 model on my Mac. The official repo is mostly useful for Windows/Linux users, but tedious for Macs using apple silicon chips. Nevertheless, please check out the [official repo](https://github.com/facebookresearch/sam3) for official documentation, I'm no expert.


## What Was Fixed

SAM3 required several fixes to run on macOS, especially Apple Silicon:

### 1. Triton GPU Kernels
- **Problem**: Triton kernels (for Euclidean Distance Transform) don't support Apple Silicon
- **Solution**: Added conditional import with OpenCV-based CPU fallback
- **File**: `sam3/model/edt.py`

### 2. Training-Only Dependencies
- **Problem**: Training modules imported packages like `decord` that don't have M1/M2/M3 builds 
- **Solution**: Made imports conditional; only loaded when needed
- **Files**: `sam3/model/sam3_tracker_base.py`, `sam3/model/sam3_image.py`

### 3. Device Handling
- **Problem**: Code hardcoded CUDA device references (this one was super annoying)
- **Solution**: Auto-detect device (CUDA if available, fallback to CPU)
- **Files**: Multiple files in `sam3/model/`

## Installation Details

### System Requirements
- **macOS 11.0+** (Monterey or later)
- **Python 3.10+** (3.12 recommended)
- **16GB+ RAM** (recommended for large images)
- **20GB+ disk space** (for model weights)

### Step 1: Create Environment
```bash
conda create -n sam3 python=3.12
conda activate sam3
```

### Step 2: Install PyTorch
For all Macs (Intel and Apple Silicon):
```bash
pip install torch torchvision torchaudio
```

PyTorch will automatically:
- Use MPS (Metal Performance Shaders) on Apple Silicon
- Use CPU with Accelerate on Intel Macs

### Step 3: Install OpenCV
```bash
# Via conda
conda install -y opencv

# Or via pip
pip install opencv-python
```

### Step 4: Install SAM3
```bash
cd /path/to/sam3
pip install -e .
```

### Step 5: Install Additional Dependencies
```bash
# For image format support (heif from iphone and stuff)
pip install pillow-heif

# For COCO evaluation
pip install pycocotools

# For system utilities
pip install psutil

```

### Step 6: HuggingFace Authentication
```bash
# Install CLI if needed
pip install huggingface-hub

# Login to HuggingFace
huggingface-cli login
```

Then enter your token from [here](https://huggingface.co/settings/tokens)

Alternatively, set token as environment variable:
```bash
export HF_TOKEN="your_token_here"
```

## Usage Examples

### Image Inference (Taken from official repo)

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load image
image = Image.open("<IMAGE_PATH>") 

# Set image and prompt
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="people")

# Get results
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(f"Found {len(masks)} objects")
```

### Visualization with OpenCV

```python
import cv2
import numpy as np

# Convert image to numpy array
image_np = np.array(image)
image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Draw bounding boxes
for box, score in zip(boxes, scores):
    x1, y1, x2, y2 = [int(c) for c in box.cpu().numpy()]
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_bgr, f"{score:.2f}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

# Save result
cv2.imwrite("output.jpg", image_bgr)
```

### Working with HEIF Images (iPhone Photos)

This isn't related to SAM3 but I thought I might as well add it here. I tried working with some photos from my iPhone for SAM3 but had some formatting issues so chatgpt gave me pillow_heif to solve this:

```python
from PIL import Image
import pillow_heif

# Register HEIF opener (do this once)
pillow_heif.register_heif_opener()

# Now you can open HEIF files directly
image = Image.open("photo_from_iphone.HEIC")
# Convert to RGB if needed
if image.mode != "RGB":
    image = image.convert("RGB")
```

### Optimization Tips
- Use smaller images (1024×1024 or less) for faster processing
- Batch processing can be slow; process one image at a time
- Use GPU-enabled machine for production inference
- For training, use NVIDIA GPU workstation

## Troubleshooting
If you ever see ModuleNotFound, the obvious thing to try would be:
```bash
  pip install <moduleName>
```
However, you may receive `No matching distribution for <ModuleName>` sometimes. In this case, you may have to try installing it with:
```bash
  conda install <moduleName>
```
If that also doesn't work, you will have to do some of the things I did, 
```bash
git grep "moduleName"
```
Go to all locations using that module and have some sort of an alternative module. Probably use chatgpt to see what it says. 

### CUDA/GPU Warnings
- Safe to ignore on macOS. PyTorch will use MPS or CPU automatically.

### Model Download Issues
- Ensure HuggingFace token is valid
- Check internet connection
- Verify disk space (20GB+ needed since model is pretty big)  

`Side Note`: I tried running the model for video segmenting and I started seeing rainbows on my cursor 😭. Highly recommend GPUs or some other additional computing power for videos.
## References

- [SAM3 GitHub Repository](https://github.com/facebookresearch/sam3)
- [HuggingFace Model Card](https://huggingface.co/facebook/sam3)


