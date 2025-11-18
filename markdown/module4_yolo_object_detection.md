---
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
# Object Detection with YOLO

This notebook demonstrates how to use the YOLO model for object detection on the IDLE-OO-Camera-Traps dataset.

First, let's install the necessary libraries.

```python
!pip install -q transformers datasets torch torchvision Pillow ultralytics scikit-learn seaborn
```

Now, let's import the required libraries.
<!-- #endregion -->

```python
import torch
from datasets import load_dataset
from PIL import Image
from IPython.display import display,HTML
import matplotlib.pyplot as plt
import requests
from ultralytics import YOLO
```

## YOLO Object Detection

First, we'll use a pretrained YOLOv8 model to perform object detection.

```python
# Load a pretrained YOLO model
model_yolo = YOLO('../yolov8n.pt')
```

Next, we load the `imageomics/IDLE-OO-Camera-Traps` dataset. We'll just take one example from the training split.

```python label="load-image-cell"
dataset = load_dataset(path="../data/IDLE-OO-Camera-Traps", split="test")
iterator = iter(dataset)
sample = next(iterator)
image = sample["image"]
display(image)
```

Now we can run the YOLO model on the image. We'll test three different confidence thresholds: 0.5, 0.1, and 0.001.

```python
confidence_thresholds = [0.5, 0.1, 0.001]

for conf in confidence_thresholds:
    print(f"Running YOLO detection with confidence threshold: {conf}")
    # Run inference on a copy of the image
    results_yolo = model_yolo(image.copy(), conf=conf)

    # Plot results
    im_array = results_yolo[0].plot()
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    display(im)
```


<!-- #region -->
### YOLO Network Architecture Analysis

To understand how the YOLO model turns image features into predictions, we can inspect its architecture. The model is generally composed of three main parts: the **Backbone**, the **Neck**, and the **Head**.

*   **Backbone**: This is a deep convolutional neural network that extracts features from the input image at various scales. YOLOv8 uses a modified CSPDarknet53 architecture.
*   **Neck**: This part connects the backbone to the head. It takes feature maps from different stages of the backbone and combines them to create richer feature pyramids. This allows the model to detect objects of different sizes more effectively. YOLOv8 uses a Path Aggregation Network (PANet) for this.
*   **Head (Detection)**: This is the final part of the network that generates the output predictions. It takes the feature maps from the neck and produces bounding boxes, class probabilities, and objectness scores.

Let's print the model structure to see the layers. We will use the pretrained `yolov8n.pt` model for this analysis.
<!-- #endregion -->

```python
# Load a pretrained YOLO model to inspect its architecture
model_to_inspect = YOLO('../yolov8n.pt')
print(model_to_inspect.model)
```

<!-- #region -->
### From Features to Predictions

The key to understanding the prediction process lies in the `Detect` module at the end of the network structure (the last layer in the printed output).

1.  **Input Feature Maps**: The `Detect` head receives feature maps from the neck at three different scales (e.g., 80x80, 40x40, 20x20 for a 640x640 input). Each scale is responsible for detecting objects of a corresponding size (small, medium, large).

2.  **Convolutional Prediction**: For each feature map, the `Detect` head applies a set of 1x1 convolutional layers. These convolutions transform the feature map's channels into a format that represents the predictions. For each location (or "patch") in the feature map, the model predicts:
    *   **Bounding Box Coordinates (4 values)**: These are typically `(x_center, y_center, width, height)`, which are regressed relative to the grid cell's location.
    *   **Class Probabilities (C values)**: A probability for each of the `C` classes the model was trained on.
    *   **Objectness Score (1 value)**: This is often implicitly part of the class prediction or a separate score indicating the confidence that an object is present in the bounding box. In YOLOv8, the box and class predictions are decoupled.

3.  **Output Tensor**: The output of the `Detect` head is a set of tensors. For a single image, the predictions from all scales are concatenated. The final output tensor has a shape like `(batch_size, num_classes + 4, num_predictions)`, where `num_predictions` is the total number of prediction anchors across all scales.

4.  **Decoding the Output**: This raw tensor output is then post-processed:
    *   The bounding box values are scaled to the original image dimensions.
    *   The class scores are passed through a softmax or sigmoid function to get final probabilities.
    *   Non-Maximum Suppression (NMS) is applied to filter out overlapping bounding boxes for the same object, keeping only the one with the highest confidence score.

This process allows the model to efficiently predict multiple objects of various sizes and classes in a single forward pass.
<!-- #endregion -->

### YOLO Intermediate Layer Visualization

Now, let's visualize the intermediate outputs of the YOLO model. We will add hooks to the model layers to capture the feature maps and then plot them.

```python
# A list to store the feature maps
feature_maps = []
hooks = []

# The hook function that saves the output of a layer
def hook_fn(module, input, output):
    feature_maps.append(output)

# We will visualize the output of the first 10 layers of the YOLO model
detection_model = model_yolo.model
layers_to_hook = detection_model.model[:10]

# Register a forward hook for each layer to be visualized
for layer in layers_to_hook:
    hooks.append(layer.register_forward_hook(hook_fn))

# Run inference to trigger the hooks.
# Make sure to clear feature_maps before running, as hooks are global.
feature_maps = []
results_yolo = model_yolo(image.copy())

# Remove the hooks after inference
for hook in hooks:
    hook.remove()

# Now, let's visualize the feature maps
from sklearn.decomposition import PCA
import numpy as np

num_layers = len(feature_maps)
layer_names = [f"Layer {i}: {type(layers_to_hook[i]).__name__}" for i in range(num_layers)]

# Plot the feature maps in a grid
cols = 4
rows = (num_layers + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()

for i, (fm, name) in enumerate(zip(feature_maps, layer_names)):
    # Detach the tensor from the computation graph and move it to the CPU
    fm = fm.detach().cpu()
    # Get the feature map for the first image in the batch
    fm = fm[0]
    
    C, H, W = fm.shape
    
    ax = axes[i]

    # Use PCA to visualize the feature map's channel dimension
    if C >= 3:
        # Reshape for PCA: from (C, H, W) to (H*W, C)
        data = fm.permute(1, 2, 0).reshape(H * W, C).numpy()
        
        # Apply PCA to reduce to 3 components for RGB visualization
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data)
        
        # Reshape back to an image (H, W, 3)
        pca_image = pca_result.reshape(H, W, 3)
        
        # Normalize each of the 3 principal components to the range [0, 1]
        for c in range(3):
            channel = pca_image[:, :, c]
            min_val, max_val = channel.min(), channel.max()
            if max_val > min_val:
                pca_image[:, :, c] = (channel - min_val) / (max_val - min_val)
            else:
                pca_image[:, :, c] = 0
        
        ax.imshow(pca_image)
        ax.set_title(name)
    else:
        # Fallback for layers with fewer than 3 channels: show the first channel in grayscale
        ax.imshow(fm[0], cmap='gray')
        ax.set_title(name + " (grayscale)")

    ax.axis('off')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
```



## Exploring the ENA24 Dataset

Instead of fine-tuning on a custom dataset, we will explore the `ENA24` dataset directly from the local checkout. This involves loading the existing labels, visualizing the frequency of common names, and displaying sample images for each unique common name.

```python
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from IPython.display import display, HTML
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define the base path to the locally checked out dataset
base_data_path = '../data/IDLE-OO-Camera-Traps/'
ena24_csv_path = os.path.join(base_data_path, 'ENA24-balanced.csv')

# Load the ENA24-balanced.csv file
ena24_df = pd.read_csv(ena24_csv_path)
print(f"Successfully loaded {ena24_csv_path}")
print(f"Total images in ENA24 dataset: {len(ena24_df)}")

# --- Visualize "common_name" frequency ---
print("\nVisualizing 'common_name' frequency...")
plt.figure(figsize=(12, 6))
ena24_df['common_name'].value_counts().plot(kind='bar')
plt.title('Frequency of Common Names in ENA24 Dataset')
plt.xlabel('Common Name')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Collect true and predicted classes for confusion matrix ---
print("\nProcessing images to collect true and predicted classes...")
y_true = []
y_pred = []

unique_common_names = ena24_df['common_name'].unique()

# Let's take a few images per class to build a more meaningful confusion matrix
num_samples_per_class = 5 

for name in unique_common_names:
    sample_images = ena24_df[ena24_df['common_name'] == name].head(num_samples_per_class)
    
    if not sample_images.empty:
        for index, row in sample_images.iterrows():
            image_relative_path = row['filepath']
            full_image_path = os.path.join(base_data_path, 'data/test/', image_relative_path)
            
            if os.path.exists(full_image_path):
                try:
                    img = Image.open(full_image_path)
                    
                    # Run YOLO detection
                    results_yolo_sample = model_yolo(img.copy(), conf=0.25)

                    y_true.append(name)

                    if len(results_yolo_sample[0].boxes) > 0:
                        # Get top prediction (results are sorted by confidence)
                        top_prediction_cls_id = int(results_yolo_sample[0].boxes.cls[0].item())
                        predicted_name = model_yolo.names[top_prediction_cls_id]
                        y_pred.append(predicted_name)
                    else:
                        y_pred.append("No detection")

                except Exception as e:
                    print(f"Could not process image {full_image_path}: {e}")
            else:
                print(f"Image file not found: {full_image_path}")

print("Finished collecting predictions.")

# --- Create and display the confusion matrix ---
print("\nGenerating confusion matrix...")

# Use pandas crosstab for a straightforward confusion matrix
df_cm = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
confusion_crosstab = pd.crosstab(df_cm['y_true'], df_cm['y_pred'], rownames=['True Class (ENA24)'], colnames=['Predicted Class (COCO)'])

plt.figure(figsize=(18, 14))
sns.heatmap(confusion_crosstab, annot=True, fmt='d', cmap='Blues')


# --- Display 1 image with YOLO detections from each unique "common_name" ---
print("\nDisplaying 1 image with YOLO detections from each unique 'common_name'...")
unique_common_names = ena24_df['common_name'].unique()

for name in unique_common_names:
    # Get 1 image for the current common name
    sample_images = ena24_df[ena24_df['common_name'] == name].head(1)
    
    if not sample_images.empty:
        print(f"\n--- Common Name: {name} ---")
        for index, row in sample_images.iterrows():
            image_relative_path = row['filepath']
            full_image_path = os.path.join(base_data_path, 'data/test/', image_relative_path)
            
            if os.path.exists(full_image_path):
                try:
                    img = Image.open(full_image_path)
                    print(f"Original Image for {name}:")
                    display(img)

                    # Run YOLO detection on the image
                    results_yolo_sample = model_yolo(img.copy(), conf=0.25) # Using a default confidence for display

                    # Plot results
                    im_array_yolo = results_yolo_sample[0].plot()
                    im_yolo = Image.fromarray(im_array_yolo[..., ::-1])  # RGB PIL image
                    print(f"YOLO Detections for {name}:")
                    display(im_yolo)
                except Exception as e:
                    print(f"Could not load or process image {full_image_path}: {e}")
            else:
                print(f"Image file not found: {full_image_path}")

```