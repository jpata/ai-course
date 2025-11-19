jupyter:
  jupytext:
    default_lexer: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Segmenting Objects with SAM using YOLO Prompts

This notebook combines our fine-tuned YOLO object detector with the Segment Anything Model (SAM) to generate precise segmentation masks for animals in the ENA24 dataset.

First, we will use our custom-trained YOLO model to predict a bounding box for an animal. Then, we will feed that bounding box as a "prompt" to SAM, which will return a high-quality segmentation mask for the object within the box.

### Motivation: Why Segment Animals?

While a bounding box tells us *where* an animal is, a segmentation mask provides a much richer understanding of the object.
*   **Precise Shape and Size**: Masks outline the exact shape of an animal, allowing for more accurate measurements of size, length, and potentially biomass estimation.
*   **Detailed Analysis**: With a precise silhouette, we can perform more detailed analyses, such as pose estimation, identifying specific body parts, or assessing animal health (e.g., whether it looks thin or well-fed).
*   **Ecological Monitoring**: Segmentation masks are crucial for large-scale ecological studies. They enable tracking individual animals across different camera trap sightings, which is essential for estimating population density, understanding territory ranges, and studying migration patterns. The precise outline helps in re-identifying individuals based on unique markings (like stripe or spot patterns).
*   **Occlusion and Crowds**: Segmentation can help distinguish between individual animals that are overlapping or close together, which is difficult with bounding boxes alone.
*   **Improved Data Quality**: Using masks instead of boxes to train downstream models (like species classifiers) can improve their accuracy by removing noisy background pixels.

This combination of a fast, specialized detector (YOLO) and a powerful, generalist segmentation model (SAM) creates an efficient and highly effective pipeline for advanced image analysis.
<!-- #endregion -->

<!-- #region -->
## 1. Setup

First, let's install the necessary libraries. `ultralytics` provides our YOLO model, and we'll install `segment-anything` for the SAM model.
<!-- #endregion -->

```python
!pip install -q ultralytics 'segment-anything' torch torchvision matplotlib opencv-python
```

<!-- #region -->
Now, let's import all the required modules.
<!-- #endregion -->

```python
import os
import torch
import requests
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

<!-- #region -->
## 2. Load Models

We need two models for this pipeline: our fine-tuned YOLO model to find the animals, and a pretrained SAM to segment them.

### 2.1. Load Fine-Tuned YOLO Model

We will load the `best.pt` weights from the latest YOLO training run performed in the `module4_yolo_finetuning.md` notebook. The code below automatically finds the latest run directory.
<!-- #endregion -->

```python
# Path to the directory where training runs are saved
train_dir = 'runs/detect'

# Find the latest training directory
train_dirs = [d for d in os.listdir(train_dir) if "train" in d]
latest_train_run = max(train_dirs, key=lambda d: os.path.getmtime(os.path.join(train_dir, d)))
best_model_path = os.path.join(train_dir, latest_train_run, 'weights/best.pt')

print(f"Loading fine-tuned YOLO model from: {best_model_path}")
yolo_model = YOLO(best_model_path)
```

<!-- #region -->
### 2.2. Load Segment Anything Model (SAM)

SAM is a foundation model from Meta AI designed for promptable image segmentation. It can generate high-quality masks from various input prompts, including points, boxes, and text.

We will use the large ViT-H SAM model. The code below will download the model checkpoint (a ~2.4GB file) if it's not already present.
<!-- #endregion -->

```python
sam_checkpoint_path = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Download the SAM checkpoint if it doesn't exist
if not os.path.exists(sam_checkpoint_path):
    print("Downloading SAM checkpoint...")
    response = requests.get(sam_checkpoint_url, stream=True)
    with open(sam_checkpoint_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")

# Load the SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
sam.to(device=device)

# Create the SAM predictor
predictor = SamPredictor(sam)
```

<!-- #region -->
## 3. Load Validation Data

We'll run our pipeline on a few images from the validation set that our YOLO model was evaluated on.
<!-- #endregion -->

```python
base_path = os.path.abspath('../data/IDLE-OO-Camera-Traps_yolo')
val_file_path = os.path.join(base_path, 'val.txt')

with open(val_file_path, 'r') as f:
    val_images = [line.strip() for line in f.readlines()]
print(f"Found {len(val_images)} validation images.")

```

<!-- #region -->
## 4. Run the YOLO-SAM Pipeline

Now we'll tie everything together. For each image, we will:
1.  Run our fine-tuned YOLO model to get bounding boxes.
2.  Take the top three highest-confidence bounding boxes as prompts.
3.  For each box, use the SAM Predictor to generate a mask.
4.  Visualize the original image, the YOLO boxes, and the final SAM masks.

First, let's define a couple of helper functions for visualization.
<!-- #endregion -->

```python
# Helper function to show a mask on the image
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Helper function to show a bounding box
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
```

<!-- #region -->
Now, let's process a few sample images and see the results.
<!-- #endregion -->

```python
# Select up to 10 images to display
num_display_images = min(len(val_images), 10)
display_images = val_images[:num_display_images]

fig, axs = plt.subplots(num_display_images, 2, figsize=(15, 5 * num_display_images))

for i, image_path_relative in enumerate(display_images):
    image_path_abs = os.path.join(base_path, image_path_relative)
    print(f"Processing: {os.path.basename(image_path_abs)}")

    # --- Load Image ---
    # SAM expects the image in RGB format
    image = cv2.imread(image_path_abs)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 1. Get YOLO Detections ---
    yolo_results = yolo_model(image_path_abs, verbose=False, conf=0.2)
    
    # Plot original image and YOLO box on the first subplot
    ax1 = axs[i, 0]
    ax1.imshow(image_rgb)
    ax1.set_title(f"YOLO Detection: {os.path.basename(image_path_abs)}")
    ax1.axis('off')

    # Plot final mask on the second subplot
    ax2 = axs[i, 1]
    ax2.imshow(image_rgb)
    ax2.set_title(f"SAM Segmentation")
    ax2.axis('off')
    
    if len(yolo_results[0].boxes) > 0:
        # Set the image for the predictor once
        predictor.set_image(image_rgb)

        # Get top 3 boxes by confidence
        boxes = yolo_results[0].boxes
        confidences = boxes.conf
        indices = torch.argsort(confidences, descending=True)
        top_indices = indices[:3]
        
        print(f"  Found {len(boxes)} objects. Segmenting top {len(top_indices)}.")

        for i in top_indices:
            box = boxes[i] # Get the box object for the current index
            box_coords = box.xyxy[0].cpu().numpy()
            
            # Draw the YOLO box on both plots for comparison
            show_box(box_coords, ax1)
            show_box(box_coords, ax2)

            # --- 2. Use Box as Prompt for SAM ---
            # The input box needs to be a numpy array
            input_box = box_coords.astype(int)

            # Predict the mask
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            # --- 3. Visualize the Mask ---
            # masks is a (1, H, W) array, so we take the first one
            show_mask(masks[0], ax2, random_color=True)
    else:
        print(f"  No objects detected by YOLO in {os.path.basename(image_path_abs)}")
        ax1.set_title(f"YOLO: No Detections in {os.path.basename(image_path_abs)}")

plt.tight_layout()
plt.show()

```

<!-- #region -->
## 5. Conclusion

In this notebook, we successfully created a powerful pipeline combining a fine-tuned YOLO model with the generalist Segment Anything Model.

*   **YOLO** acted as a fast and efficient "object finder," leveraging its specialized training on our ENA24 dataset to locate animals with high accuracy.
*   **SAM** acted as a "precision tool," taking the rough bounding box from YOLO and producing a highly detailed and accurate segmentation mask without needing to be trained on our specific data.

This approach demonstrates the power of foundation models in modern computer vision. Instead of training a complex segmentation model from scratch, which can be data-intensive and time-consuming, we can link together smaller, more specialized components and large, pre-trained models to achieve state-of-the-art results with significantly less effort. This YOLO-SAM pipeline could be used to rapidly generate a high-quality segmentation dataset, which could then be used to train a more lightweight, custom segmentation model if real-time performance on-device were a final goal.
<!-- #endregion -->
