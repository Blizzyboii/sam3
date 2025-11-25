# Most of this code can be found in the official repository
# The only few changes here are usage of cv2 to draw bounding boxes and save images
## BELOW IS THE CODE FOR WORKING WITH VIDEOS
'''from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "sample.mp4" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="capture all people",
    )
)
output = response["outputs"]'''

# BELOW IS THE CODE FOR WORKING WITH IMAGES
import torch
import cv2
import numpy as np

from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
image_path = "random.jpeg"
image = Image.open(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image from {image_path}")
# Convert to RGB if needed
if image.mode != "RGB":
    image = image.convert("RGB")

# Set the image for processing
inference_state = processor.set_image(image)

# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="people")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

if len(scores) > 0:
    print(f"Top score: {scores[0]:.4f}")

# Convert PIL image to numpy array for OpenCV
image_np = np.array(image)
# Convert RGB to BGR for OpenCV (already in BGR from cv2 read)
image_bgr_output = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Draw bounding boxes on the image
for i, (box, score) in enumerate(zip(boxes, scores)):
    # boxes are in (x1, y1, x2, y2) format (already in absolute coordinates)
    box_coords = box.cpu().numpy()
    x1, y1, x2, y2 = int(box_coords[0]), int(box_coords[1]), int(box_coords[2]), int(box_coords[3])
    
    # Draw rectangle
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2
    cv2.rectangle(image_bgr_output, (x1, y1), (x2, y2), color, thickness)
    
    # Add text with score
    text = f"people: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0, 255, 0)  # Green
    cv2.putText(image_bgr_output, text, (x1, y1 - 10), font, font_scale, text_color, font_thickness)
    
    print(f" score: {score:.4f}")

# Save the image with bounding boxes
output_path = "output.jpg"
cv2.imwrite(output_path, image_bgr_output)