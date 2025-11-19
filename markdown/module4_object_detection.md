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
# Object Detection: A Comparison of YOLO and DETR

This notebook introduces and compares two prominent object detection architectures: YOLO (You Only Look Once) and DETR (Real-Time DEtection TRansformer). We will use pretrained models to perform inference on the ENA24 dataset and analyze their performance, paying special attention to the challenges posed by the mismatch between the models' training classes (COCO) and the dataset's actual classes.

Object detection is a computer vision task that involves identifying and locating objects within an image. A model performing this task returns a set of bounding boxes, each with a corresponding class label for the object it contains.

We will explore:
*   **YOLO**: A leading family of single-stage detectors known for its speed and efficiency.
*   **DETR**: A modern, transformer-based, end-to-end detector that provides high accuracy without requiring complex post-processing steps like Non-Maximum Suppression (NMS).
*   **The ENA24 Dataset**: We will use the `imageomics/IDLE-OO-Camera-Traps` dataset to evaluate how well these models, pretrained on general-purpose datasets, perform on specialized data.
<!-- #endregion -->

<!-- #region -->
## 1. Setup

First, let's install the necessary libraries. `ultralytics` provides the YOLO model, while `transformers` gives us access to DETR.
<!-- #endregion -->

```python
!pip install -q ultralytics transformers timm datasets torch torchvision Pillow scikit-learn seaborn pandas matplotlib
```

<!-- #region -->
Now, let's import all the required modules.
<!-- #endregion -->

```python
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import requests
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import cv2

# YOLO imports
from ultralytics import YOLO

# DETR imports
from transformers import DetrImageProcessor, DetrForObjectDetection
```

<!-- #region -->
## 2. Loading the Dataset and a Sample Image

We'll load the `imageomics/IDLE-OO-Camera-Traps` dataset from a local path and select one example from the test split to use for our initial inference examples.
<!-- #endregion -->

```python
image_path = "../data/IDLE-OO-Camera-Traps/data/test/desert-lion/8b0146e9-3117-4d76-b61c-a8ead22e5755.png"
image = Image.open(image_path).convert("RGB")
print(f"Loaded image: {image_path}")
display(image)
```

<!-- #region -->
## 3. Part 1: YOLO (You Only Look Once)

The YOLO family of models are "single-stage" detectors, meaning they predict bounding boxes and class probabilities directly from the image in a single pass. They are famously fast and have become an industry standard for real-time object detection.

### 3.1. Inference with YOLOv8

The `ultralytics` library makes it incredibly simple to load a pretrained YOLO model and run inference. We'll use `yolov8n.pt`, a nano-sized version of the model that is fast and lightweight.

The model is pretrained on the COCO dataset, a large-scale object detection dataset with 80 common object classes like "person," "car," and "dog."
<!-- #endregion -->

```python
# Load a pretrained YOLOv8 model
model_yolo = YOLO('../yolov8n.pt')

# Run inference on a copy of the image
results_yolo = model_yolo(image.copy(), conf=0.5)

# The `plot()` method conveniently draws the detected boxes on the image
im_array = results_yolo[0].plot()
im_yolo = Image.fromarray(im_array[..., ::-1])  # Convert to RGB PIL image

print("YOLOv8 Detections (Confidence > 0.5):")
display(im_yolo)
```

<!-- #region -->
### 3.2. YOLO Architectural Deep Dive

YOLO's architecture is a masterclass in efficiency, designed to perform detection in a single forward pass. It consists of three primary components: the Backbone, the Neck, and the Head.

*   **Backbone (CSPDarknet):** The backbone is a deep Convolutional Neural Network (CNN) responsible for extracting features from the input image at various scales. It starts with a `Stem` layer for initial downsampling, followed by a series of convolutional blocks (`C2f` in YOLOv8). As the image passes through the backbone, its spatial dimensions (height and width) are reduced, while the number of channels (feature depth) is increased. This process creates a hierarchy of feature maps: early layers capture low-level features like edges and textures, while deeper layers capture high-level semantic features like object parts.

*   **Neck (PANet):** The neck's job is to fuse the feature maps from the backbone to create a feature pyramid that is rich in both semantic (what) and localization (where) information. YOLOv8 uses a Path Aggregation Network (PANet). It takes feature maps from different stages of the backbone and combines them through both a top-down path (bringing high-level context to lower-level maps) and a bottom-up path (bringing precise localization information from lower-level maps to higher-level ones). This allows the model to effectively detect objects of different sizes.

*   **Head (YOLOv8 Head):** The head is the final stage, responsible for making predictions. It takes the fused feature maps from the neck and uses a series of convolutions to predict three things for each location on the feature grid:
    1.  **Bounding Box:** The coordinates (x, y, width, height) of a potential object.
    2.  **Objectness Score:** A confidence score indicating how likely it is that an object exists at this location.
    3.  **Class Probabilities:** A set of probabilities for each of the 80 COCO classes.

#### Output Interpretation and NMS

The raw output of the YOLO head is a massive tensor containing thousands of potential detections at different scales. To produce a clean, final list of objects, a critical post-processing step is required: **Non-Maximum Suppression (NMS)**. NMS works by:
1.  Filtering out boxes with low confidence scores.
2.  For each class, finding groups of overlapping boxes that likely correspond to the same object.
3.  Within each group, suppressing (discarding) all boxes except the one with the highest confidence score.

The `ultralytics` library handles all of this automatically when you call the model.

#### Visualizing Intermediate Features

To better understand what the model "sees," we can extract the feature maps from intermediate layers and visualize them. We will use Principal Component Analysis (PCA) to reduce the high-dimensional channel information of a feature map into 3 components (RGB) for visualization.

We'll grab features from three different points in the network:
1.  An early backbone layer (`C2f_2`)
2.  A later backbone layer (`C2f_4`)
3.  The output of a neck layer (`C2f_6`)
<!-- #endregion -->

```python
# Helper function to visualize a feature map using PCA
def visualize_features_pca(feature_map, title):
    # Detach from graph and move to CPU
    features = feature_map.squeeze(0).cpu().numpy()
    
    # Reshape for PCA: (H*W, C)
    # The input shape is (C, H, W), so we transpose it to (H, W, C) first
    print("feat", features.shape)
    features = features.transpose(1, 2, 0)
    h, w, c = features.shape
    reshaped_features = features.reshape(-1, c)
    
    # Apply PCA to reduce channels to 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped_features)
    
    # Normalize and reshape back to an image (H, W, 3)
    pca_img = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    pca_img = pca_img.reshape(h, w, 3)
    
    # Display
    plt.figure(figsize=(6, 6))
    plt.imshow(pca_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- YOLO Feature Extraction ---

# Dictionary to store intermediate features
yolo_features = {}

# Hook function to capture the output of a module
def get_yolo_hook(name):
    def hook(model, input, output):
        # For C2f modules, the output might be a tuple/list, we take the first tensor
        if isinstance(output, (list, tuple)):
            yolo_features[name] = output[0].detach()
        else:
            yolo_features[name] = output.detach()
    return hook

# Mapping friendly names to actual module names in the YOLOv8 model structure
# The indices [2], [4], [6] correspond to early backbone, late backbone, and neck layers
yolo_layer_map = {
    "Early Backbone (C2f_2)": model_yolo.model.model[2],
    "Mid Backbone (C2f_4)": model_yolo.model.model[4],
    "Neck Features (C2f_6)": model_yolo.model.model[6],
}

# Register forward hooks on the target layers
hooks = []
for name, layer in yolo_layer_map.items():
    hooks.append(layer.register_forward_hook(get_yolo_hook(name)))

# Run inference on the sample image to trigger the hooks
# We use the original `image` from the dataset
results_yolo = model_yolo(image.copy(), verbose=False)

# Remove the hooks now that we have the features
for hook in hooks:
    hook.remove()

# Visualize the captured features
print("PCA Visualization of YOLOv8 Intermediate Features:")
for name, features in yolo_features.items():
    visualize_features_pca(features, name)
```

<!-- #region -->
## 4. Part 2: DETR (DEtection TRansformer)

DETR (DEtection TRansformer) models reframe object detection as a direct set prediction problem. They use a transformer-based architecture to produce a fixed-size set of predictions, eliminating the need for complex post-processing like NMS. DETR is an evolution of this idea, optimized for real-time performance.

### 4.1. Inference with DETR

We will use the `transformers` library to load a pretrained DETR model. Unlike YOLO, DETR models require a specific `processor` to resize and normalize the input image correctly.
<!-- #endregion -->

```python
# Load the processor and a pretrained DETR model from Hugging Face
processor_detr = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Prepare the image for the model
inputs = processor_detr(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model_detr(**inputs)

# Post-process the results to get bounding boxes and class labels
# We set a low threshold to get all potential detections, then we'll select the top 3.
target_sizes = torch.tensor([image.size[::-1]])
results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]

# Helper function to draw bounding boxes
def draw_boxes(image, boxes, labels, scores):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    for box, label, score in zip(boxes, labels, scores):
        box = [round(i, 2) for i in box.tolist()]
        label_text = f"{label} {score:.2f}"
        
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), label_text, fill="red")
    return img_draw

# Get labels, scores, and boxes for the top 3 detections
scores = results_detr["scores"]
labels = results_detr["labels"]
boxes = results_detr["boxes"]

TOP_K = 3
k = min(len(scores), TOP_K)

scores, indices = torch.topk(scores, k)
labels = labels[indices]
boxes = boxes[indices]

# Convert label IDs to names
labels = [model_detr.config.id2label[i.item()] for i in labels]

# Draw the boxes on the image
im_detr = draw_boxes(image, boxes, labels, scores)

print(f"DETR Top {k} Detections:")
display(im_detr)
```

<!-- #region -->
### 4.2. DETR Architectural Deep Dive

The DETR architecture introduced a paradigm shift by framing object detection as a direct set prediction problem, removing the need for many hand-designed components like NMS.

*   **Backbone (ResNet):** It begins with a standard CNN backbone (a ResNet-50 in this case) to extract a 2D feature map from the input image. This feature map captures the essential spatial features.

*   **Transformer Encoder:** This is where DETR diverges significantly.
    *   **Input:** The feature map from the backbone is flattened into a sequence of tokens. Crucially, these tokens are combined with **Positional Encodings**, which are vectors that give the model information about the original `(x, y)` position of each token. Without this, the transformer would be unaware of the image's spatial structure.
    *   **Function:** The encoder processes this sequence using multiple layers of self-attention. This allows every feature token to attend to every other token, building a rich, context-aware representation. The output is an enriched sequence of image features.

*   **Transformer Decoder:** The decoder is the core of the prediction mechanism.
    *   **Input:** It takes two main inputs: the memory of enriched features from the encoder, and a small, fixed-size set of learnable embeddings called **Object Queries**.
    *   **Function:** Each object query acts as a "slot" responsible for detecting a single object. Through layers of self-attention and cross-attention, the queries interact with each other (to avoid duplicates) and with the encoder's output (to find and localize objects). Each query "asks" the image features: "Is there an object here that matches my pattern?"
    
*   **Prediction Heads (FFNs):** After the final decoder layer, each output query embedding is passed to two separate Feed-Forward Networks (FFNs):
    1.  A **classification head** predicts the class label for that query (e.g., 'bird', 'car', or 'no object').
    2.  A **box head** predicts the bounding box coordinates `(center_x, center_y, width, height)`.

#### End-to-End Philosophy

This design is "end-to-end" because it directly outputs a sparse set of predictions. Since each query is encouraged to specialize on a different object, the model learns to avoid making duplicate predictions for the same object, thus eliminating the need for NMS.

<!-- #endregion -->

#### Visualizing Attention Maps

```python
# --- STEP 1: INFERENCE WITH ATTENTIONS ---
# We must request output_attentions=True to get the internal weights
print("Running inference with attentions...")
with torch.no_grad():
    outputs = model_detr(**inputs, output_attentions=True)
    
    # 1. Get Probabilities and Top K
    probs = outputs.logits.softmax(-1)[0, :, :-1]
    max_scores, class_ids = probs.max(-1)
    TOP_K = 3
    top_scores, top_idxs = torch.topk(max_scores, TOP_K)

# --- STEP 2: PROCESS ATTENTION MAPS ---
print("Generating Cross-Attention Maps...")

# Get the Cross-Attentions from the LAST decoder layer
# Shape: (Batch, Num_Heads, Num_Queries, Sequence_Length)
# Sequence_Length = Feature_Height * Feature_Width (usually H/32 * W/32)
cross_attentions = outputs.cross_attentions[-1]

# Get input image dimensions
img_h, img_w = inputs['pixel_values'].shape[-2:]

# Prepare background image for visualization
img_tensor = inputs['pixel_values'][0].detach().cpu()
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
rgb_img_base = (img_tensor * std + mean).permute(1, 2, 0).numpy()
rgb_img_base = np.clip(rgb_img_base, 0, 1)

# Setup Plot
fig, axes = plt.subplots(1, TOP_K, figsize=(20, 6))
if TOP_K == 1: axes = [axes]

for i, query_idx_tensor in enumerate(top_idxs):
    query_idx = query_idx_tensor.item()
    class_id = class_ids[query_idx].item()
    score = top_scores[i].item()
    label_name = model_detr.config.id2label[class_id]

    print(f"Rank {i+1}: {label_name} | Score: {score:.2f} | Query Index: {query_idx}")

    # 1. Extract Attention for this specific Query
    # Shape: [Batch, Heads, Queries, Seq] -> [Heads, Seq]
    attn_map = cross_attentions[0, :, query_idx, :]
    
    # 2. Average over Attention Heads
    # Shape: [Seq]
    attn_map = attn_map.mean(dim=0).detach().cpu()

    # 3. Reshape Sequence back to 2D Feature Map
    # We infer the feature map size (H/32, W/32)
    feat_h = int(np.ceil(img_h / 32))
    feat_w = int(np.ceil(img_w / 32))
    
    # Handle potential mismatch due to padding/rounding
    num_patches = feat_h * feat_w
    # DETR sometimes pads the mask, so we strictly reshape to what matches the sequence length
    # If exact match isn't possible, we calculate approximate square side
    if attn_map.shape[0] != num_patches:
       # Fallback for edge cases: assume roughly square aspect ratio
       side = int(attn_map.shape[0]**0.5)
       feat_h, feat_w = side, side

    attn_map = attn_map.reshape(feat_h, feat_w).numpy()

    # 4. Process for Visualization
    # Resize to original image size
    attn_map = cv2.resize(attn_map, (img_w, img_h))
    
    # Normalize 0-1
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # Apply ColorMap
    heatmap = np.uint8(255 * attn_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    visualization = cv2.addWeighted(np.uint8(255 * rgb_img_base), 0.6, heatmap, 0.4, 0)

    # 5. Draw Box
    pred_box = outputs.pred_boxes[0, query_idx].detach().cpu().numpy()
    cx, cy, w_box, h_box = pred_box
    x1 = int((cx - w_box / 2) * img_w)
    y1 = int((cy - h_box / 2) * img_h)
    x2 = int((cx + w_box / 2) * img_w)
    y2 = int((cy + h_box / 2) * img_h)

    cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Label
    label_text = f"{label_name} {score:.2f}"
    cv2.putText(visualization, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    axes[i].imshow(visualization)
    axes[i].axis('off')
    axes[i].set_title(f"Rank {i+1}: {label_name}", fontsize=14)

plt.tight_layout()
plt.show()
```

<!-- #region -->
## 5. Comparative Analysis on the ENA24 Dataset

Both models are powerful, but how do they perform on a real-world, specialized dataset like ENA24? A key challenge is that they were trained on COCO's 80 classes, which do not directly align with the animal species in ENA24.

### 5.1. The Class Mismatch Problem

Let's look at the classes in the ENA24 dataset. We expect the models to either fail to detect anything or to classify an ENA24 animal as a "related" COCO class (e.g., classifying a "deer" as a "cow" or "horse").
<!-- #endregion -->

```python
# Define the base path and load the ENA24 metadata
base_data_path = '../data/IDLE-OO-Camera-Traps/'
ena24_csv_path = os.path.join(base_data_path, 'ENA24-balanced.csv')
ena24_df = pd.read_csv(ena24_csv_path)

# --- Visualize "common_name" frequency ---
print("Visualizing 'common_name' frequency in the ENA24 Dataset:")
plt.figure(figsize=(12, 6))
ena24_df['common_name'].value_counts().plot(kind='bar')
plt.title('Frequency of Common Names in ENA24 Dataset')
plt.xlabel('Common Name')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

<!-- #region -->
### 5.2. Performance Evaluation

We will now iterate through a sample of the ENA24 dataset, run both models on each image, and record the ground truth class and the top predicted COCO class. This will allow us to see which COCO classes the models associate with the ENA24 animals.
<!-- #endregion -->

```python
y_true = []
y_pred_yolo = []
y_pred_detr = []

# Let's take a few images per class for our analysis
num_samples_per_class = 5
unique_common_names = ena24_df['common_name'].unique()

for name in unique_common_names:
    sample_images = ena24_df[ena24_df['common_name'] == name].head(num_samples_per_class)
    
    for index, row in sample_images.iterrows():
        image_relative_path = row['filepath']
        full_image_path = os.path.join(base_data_path, 'data/test/', image_relative_path)
        
        if os.path.exists(full_image_path):
            try:
                img = Image.open(full_image_path).convert("RGB")
                y_true.append(name)

                # --- Run YOLO detection ---
                results_yolo = model_yolo(img.copy(), conf=0.25, verbose=False)
                if len(results_yolo[0].boxes) > 0:
                    top_pred_id = int(results_yolo[0].boxes.cls[0].item())
                    predicted_name = model_yolo.names[top_pred_id]
                    y_pred_yolo.append(predicted_name)
                else:
                    y_pred_yolo.append("No detection")

                # --- Run DETR detection ---
                inputs = processor_detr(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = model_detr(**inputs)
                
                target_sizes = torch.tensor([img.size[::-1]])
                results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.25)[0]
                
                if len(results_detr["scores"]) > 0:
                    top_pred_id = results_detr["labels"][0].item()
                    predicted_name = model_detr.config.id2label[top_pred_id]
                    y_pred_detr.append(predicted_name)
                else:
                    y_pred_detr.append("No detection")

            except Exception as e:
                print(f"Could not process image {full_image_path}: {e}")

print("Finished collecting predictions.")
```

<!-- #region -->
### 5.3. Visualization with a Confusion Matrix

A standard confusion matrix compares predictions against true labels when the classes are the same. Since our classes are different, we'll use a **cross-tabulation** (crosstab) to visualize the relationship between the true ENA24 classes and the predicted COCO classes from each model.
<!-- #endregion -->

```python
# --- YOLO Crosstab ---
df_yolo = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred_yolo})
yolo_crosstab = pd.crosstab(df_yolo['y_true'], df_yolo['y_pred'], rownames=['True Class (ENA24)'], colnames=['Predicted Class (YOLO/COCO)'])

plt.figure(figsize=(18, 14))
sns.heatmap(yolo_crosstab, annot=True, fmt='d', cmap='Blues')
plt.title('ENA24 True Class vs. YOLO Predicted COCO Class')
plt.show()

# --- DETR Crosstab ---
df_detr = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred_detr})
detr_crosstab = pd.crosstab(df_detr['y_true'], df_detr['y_pred'], rownames=['True Class (ENA24)'], colnames=['Predicted Class (DETR/COCO)'])

plt.figure(figsize=(18, 14))
sns.heatmap(detr_crosstab, annot=True, fmt='d', cmap='Greens')
plt.title('ENA24 True Class vs. DETR Predicted COCO Class')
plt.show()
```

<!-- #region -->
From the heatmaps, we can see how the models perform. For example, both models often correctly map the ENA24 `bird` class to the COCO `bird` class. However, for species not in COCO, like `red_deer` or `wild_boar`, the models might predict related classes like `cow`, `horse`, or `dog`, or simply fail to make a detection. This analysis highlights the limitations of using pretrained models on novel domains and underscores the need for fine-tuning.
<!-- #endregion -->

<!-- #region -->
## 6. Head-to-Head Comparison: YOLO vs. DETR

| Feature               | YOLOv8                                        | DETR                                                       |
| --------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| **Architecture**      | CNN-based (CSPDarknet Backbone, PANet Neck)   | Hybrid (CNN Backbone + Transformer Encoder/Decoder)        |
| **Prediction**        | Predicts on a dense grid across the image     | Predicts a sparse set of objects via object queries        |
| **Post-processing**   | Requires Non-Maximum Suppression (NMS)        | End-to-end; minimal or no NMS required                     |
| **Ease of Use**       | Very simple via `ultralytics` library         | More complex; requires manual processing via `transformers`|
| **Core Idea**         | Fast, single-stage regression and classification | Direct set prediction, treating detection as a dictionary lookup |
<!-- #endregion -->

<!-- #region -->
## 7. Conclusion

In this notebook, we explored two state-of-the-art object detection models.
*   **YOLOv8** is incredibly fast and easy to use, making it an excellent choice for real-time applications where speed is critical. Its reliance on NMS is a defining characteristic of single-stage detectors.
*   **DETR** represents a newer paradigm, using a transformer architecture to perform end-to-end detection. This removes the need for hand-tuned components like NMS and can lead to better performance, though often at the cost of higher computational requirements and implementation complexity.

Our analysis on the ENA24 dataset showed that while pretrained models are a fantastic starting point, their performance on specialized domains is limited by their training data. To achieve high accuracy on ENA24's specific animal classes, the clear next step is **fine-tuning**, where we would train these models further on the ENA24 data itself.
<!-- #endregion -->
