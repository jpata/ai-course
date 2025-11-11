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
dataset = load_dataset("imageomics/IDLE-OO-Camera-Traps", split="test", streaming=True)
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
