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
!pip install -q transformers datasets torch torchvision Pillow ultralytics scikit-learn
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
model_yolo = YOLO('yolov8n.pt')
```

Next, we load the `imageomics/IDLE-OO-Camera-Traps` dataset. We'll just take one example from the training split.

```python label="load-image-cell"
dataset = load_dataset(path="./data/IDLE-OO-Camera-Traps", split="test")
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

# Define the base path to the locally checked out dataset
base_data_path = 'data/IDLE-OO-Camera-Traps/'
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

# --- Display 1 image from each unique "common_name" ---
print("\nDisplaying 1 image from each unique 'common_name'...")
unique_common_names = ena24_df['common_name'].unique()

for name in unique_common_names:
    # Get up to 2 image paths for the current common name
    sample_images = ena24_df[ena24_df['common_name'] == name].head(1)
    
    if not sample_images.empty:
        print(f"\n--- Common Name: {name} ---")
        for index, row in sample_images.iterrows():
            # Construct the full image path
            # The 'filepath' column contains paths like 'ENA24/image_uuid.png'
            # The actual images are under data/IDLE-OO-Camera-Traps/data/test/ENA24/
            image_relative_path = row['filepath']
            # Assuming the images are in data/IDLE-OO-Camera-Traps/data/test/
            full_image_path = os.path.join(base_data_path, 'data/test/', image_relative_path)
            
            if os.path.exists(full_image_path):
                try:
                    img = Image.open(full_image_path)
                    display(img)

                    # Run YOLO detection on the image
                    results_yolo_sample = model_yolo(img.copy(), conf=0.25) # Using a default confidence for display

                    # Plot results
                    im_array_yolo = results_yolo_sample[0].plot()
                    im_yolo = Image.fromarray(im_array_yolo[..., ::-1])  # RGB PIL image
                    print(f"YOLO Detections for {name}:")
                    display(im_yolo)
                except Exception as e:
                    print(f"Could not load image {full_image_path}: {e}")
            else:
                print(f"Image file not found: {full_image_path}")
```