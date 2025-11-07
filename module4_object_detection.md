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
# Object Detection with YOLO and OWL2

This notebook demonstrates how to use the YOLO and OWL2 models for object detection on the IDLE-OO-Camera-Traps dataset.

First, let's install the necessary libraries.

```python
!pip install -q transformers datasets torch torchvision Pillow ultralytics scikit-learn
```

Now, let's import the required libraries.
<!-- #endregion -->

```python
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from datasets import load_dataset
from PIL import Image, ImageDraw
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

## Object Detection with OWL2

Now, we will use the OWL2 model for object detection.

We will load the OWL2 model and processor from Hugging Face.

```python
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
```

Now, let's define the objects we want to detect. We can see a deer in the image, so let's try to detect that.

```python
texts = [["a photo of a leopard", "a photo of a tiger", "a photo of a rock"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
```

Now we run the model to get the object detection outputs.

```python
with torch.no_grad():
  outputs = model(**inputs)
```


The model outputs logits and bounding boxes. We need to post-process these to visualize them.

```python
# Get predictions
target_sizes = torch.Tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)

i = 0  # Retrieve predictions for the first image
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Draw bounding boxes on a copy of the image
image_with_boxes = image.copy()
draw = ImageDraw.Draw(image_with_boxes)

# Define a list of colors to use for different labels
colors = ["red", "green", "blue", "yellow", "purple", "orange"]

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    # Assign a color based on the label
    color = colors[label.item() % len(colors)]
    print(
        f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
    )
    draw.rectangle(box, outline=color, width=3)
    # Draw the label and confidence score
    draw.text((box[0], box[1]), f"{text[label]} {round(score.item(), 3)}", fill=color)

image_with_boxes
```

### OWL Intermediate Layer Visualization

Similar to YOLO, we can visualize the intermediate layers of the OWL's vision model. The OWL model uses a Vision Transformer (ViT), so the intermediate features are sequences of patch embeddings. We'll reshape them back into a grid and use PCA to visualize.

```python
# A list to store the feature maps
feature_maps = []
hooks = []

# The hook function that saves the output of a layer
def hook_fn_owl(module, input, output):
    # The output of Owlv2EncoderLayer is a tuple, we're interested in the hidden states
    feature_maps.append(output[0])

# We will visualize the output of a few layers from the vision encoder
vision_encoder_layers = model.owlv2.vision_model.encoder.layers
# Let's pick the first, a middle, and the last layer of the 12-layer encoder
layers_to_hook_indices = [0, 5, 11]
layers_to_hook = [vision_encoder_layers[i] for i in layers_to_hook_indices]

# Register a forward hook for each layer
for layer in layers_to_hook:
    hooks.append(layer.register_forward_hook(hook_fn_owl))

# Run inference to trigger the hooks
feature_maps = []
with torch.no_grad():
    outputs_viz = model(**inputs)

# Remove the hooks
for hook in hooks:
    hook.remove()

from sklearn.preprocessing import StandardScaler

# Now, let's visualize the feature maps
num_layers = len(feature_maps)
layer_names = [f"Layer {i}" for i in layers_to_hook_indices]

# Get the patch grid dimensions
_, _, H_proc, W_proc = inputs['pixel_values'].shape
patch_size = model.owlv2.vision_model.config.patch_size
h_patches = H_proc // patch_size
w_patches = W_proc // patch_size

# Plot the feature maps in a grid
cols = num_layers
rows = 1
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
if num_layers == 1:
    axes = [axes]
axes = axes.flatten()

for i, (fm, name) in enumerate(zip(feature_maps, layer_names)):
    fm = fm.detach().cpu()[0] # Get batch 0
    
    num_patches = fm.shape[0]
    expected_patches = h_patches * w_patches
    
    ax = axes[i]
    
    data_to_visualize = None
    # Check if there is an extra token (like a CLS token)
    if num_patches == expected_patches + 1:
        # Exclude the first token (CLS token)
        data_to_visualize = fm[1:].numpy()
    elif num_patches == expected_patches:
        data_to_visualize = fm.numpy()

    if data_to_visualize is not None:
        # It's a good practice to scale data before PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_visualize)
        
        # Apply PCA to visualize the patch embeddings
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_data)
        
        # Reshape back to an image
        pca_image = pca_result.reshape(h_patches, w_patches, 3)
        
        # Normalize for display
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
        # If shape is still unexpected, show an error
        ax.text(0.5, 0.5, f"Shape mismatch:\n{num_patches} patches vs\n{expected_patches} expected", ha='center', va='center')
        ax.set_title(name)

    ax.axis('off')

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
```

### OWL Text-Vision Attention Visualization

To understand how the text prompts guide the object detection, we can visualize the cross-attention scores between the text query embeddings and the image patch embeddings from the vision transformer's final layer. This shows which parts of the image the model focuses on for a given text query.

```python
# Run the model with output_attentions=True to get the attention scores
with torch.no_grad():
    outputs_attn = model(**inputs, output_attentions=True)

# The cross-attention weights are in the 'cross_attentions' attribute of the output.
# This is a tuple with one element per decoder layer in the prediction head. We'll take the last one.
cross_attentions = outputs_attn.cross_attentions[-1] # Last layer of the decoder

# The shape is (batch_size, num_heads, num_queries, sequence_length)
# num_queries corresponds to the text tokens, and sequence_length to the image patches.
# We average the attention scores across all attention heads.
attention_scores = cross_attentions.mean(dim=1)[0] # batch 0

# Get the patch grid dimensions
_, _, H_proc, W_proc = inputs['pixel_values'].shape
patch_size = model.owlv2.vision_model.config.patch_size
h_patches = H_proc // patch_size
w_patches = W_proc // patch_size

# The attention scores are for each token. Let's average the attention for tokens within each prompt.
# The processor concatenates the prompts, separated by [SEP] tokens.
sep_token_id = processor.tokenizer.sep_token_id
sep_indices = (inputs.input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0].tolist()

prompt_attentions = []
start_idx = 1 # After [CLS]
for sep_idx in sep_indices:
    # Don't include the [SEP] token itself in the average
    if sep_idx > start_idx:
        prompt_attentions.append(attention_scores[start_idx:sep_idx].mean(dim=0))
    start_idx = sep_idx + 1

# Number of text prompts
num_prompts = len(texts[0])

fig, axes = plt.subplots(1, num_prompts, figsize=(20, 5))
if num_prompts == 1:
    axes = [axes]

for i, (prompt_text, attention_map) in enumerate(zip(texts[0], prompt_attentions)):
    ax = axes[i]
    
    # Reshape the attention map to the patch grid
    attention_heatmap = attention_map.reshape(h_patches, w_patches).cpu().numpy()
    
    # Plot the original image
    ax.imshow(image)
    
    # Overlay the heatmap
    # We use extent to scale the heatmap to the image size
    ax.imshow(attention_heatmap, cmap='jet', alpha=0.5, extent=(0, image.width, image.height, 0))
    
    ax.set_title(f"Attention for: '{prompt_text}'")
    ax.axis('off')

plt.tight_layout()
plt.show()
```
