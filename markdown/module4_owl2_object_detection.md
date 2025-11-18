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
# Object Detection with OWL2

This notebook demonstrates how to use the OWL2 model for object detection on the IDLE-OO-Camera-Traps dataset.

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
import tqdm
import numpy as np
```

Next, we load the `imageomics/IDLE-OO-Camera-Traps` dataset. We'll just take one example from the training split.

```python label="load-image-cell"
dataset = load_dataset(path="../data/IDLE-OO-Camera-Traps", split="test")
iterator = iter(dataset)
sample = next(iterator)
print(sample)
image = sample["image"]
display(image)
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

## Investigating OWL2 Internals

OWL2 predicts for each patch whether or not it contains an object, and its bounding box.
It also gives an agnostic class embedding for each patch, which contains information about the contents of the box.

```python
feature_map = model.image_embedder(inputs.pixel_values)[0]
batch_size, height, width, hidden_size = feature_map.shape
image_features = feature_map.reshape(batch_size, height * width, hidden_size)
source_class_embeddings = model.class_predictor(image_features)[1]
objectnesses = model.objectness_predictor(image_features).sigmoid()
boxes = model.box_predictor(image_features, feature_map=feature_map)

source_class_embeddings.shape
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
num_patches = model.config.vision_config.image_size // model.config.vision_config.patch_size
H, W = num_patches, num_patches
pca_result = pca.fit_transform(source_class_embeddings[0].detach().numpy())
pca_image = pca_result.reshape(H, W, 3)
for c in range(3):
    channel = pca_image[:, :, c]
    min_val, max_val = channel.min(), channel.max()
    if max_val > min_val:
        pca_image[:, :, c] = (channel - min_val) / (max_val - min_val)
    else:
        pca_image[:, :, c] = 0
```

```python
plt.title("Class embeddings")
plt.imshow(pca_image)
```

```python
plt.title("Objectness scores")
plt.imshow(objectnesses.detach().numpy().reshape(H,W))
```

```python
def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(processor.image_processor.image_std)[:, None, None]) + np.array(processor.image_processor.image_mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image
```

```python
top_scores = np.argsort(objectnesses.detach().numpy()[0])[-3:]
# Plot the original image, and the boxes of the top 3 objects.
plt.figure(figsize=(10, 10))
img_preprocessed = get_preprocessed_image(inputs.pixel_values)
plt.imshow(img_preprocessed)
plt.title("Original Image with Top 3 Objectness Boxes (Raw Model Output)")
plt.axis('off')

# Create a drawing object on a copy of the image to avoid modifying the original
image_with_raw_boxes = img_preprocessed.copy()
draw = ImageDraw.Draw(image_with_raw_boxes)

img_width, img_height = img_preprocessed.size

# Iterate over the top 3 objectness scores and their corresponding boxes
for patch_idx in top_scores:
    # Get the raw box coordinates for this patch
    raw_box = boxes[0, patch_idx].detach().numpy()

    # Assuming raw_box is [x_min_norm, y_min_norm, x_max_norm, y_max_norm] relative to feature map (0-1)
    # Scale to original image dimensions
    cx, cy, w, h = raw_box
    cx = cx * img_width
    cy = cy * img_height
    w = w * img_width
    h = h * img_height

    box_coords_pixel = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

    # Draw the rectangle
    draw.rectangle(box_coords_pixel, outline="lime", width=3)

plt.imshow(image_with_raw_boxes)
```

## Automatic data labelling

Here we will loop over images from the ENA24 dataset and apply prompts to automatically label them.

```python
import pandas as pd
import os
import shutil

device = torch.device("cuda")
model.to(device)

# Define the base path to the locally checked out dataset
base_data_path = '../data/IDLE-OO-Camera-Traps/'
base_data_path_yolo = '../data/IDLE-OO-Camera-Traps_yolo'
ena24_csv_path = os.path.join(base_data_path, 'ENA24-balanced.csv')

# Load the ENA24-balanced.csv file
ena24_df = pd.read_csv(ena24_csv_path)
print(f"Successfully loaded {ena24_csv_path}")
print(f"Total images in ENA24 dataset: {len(ena24_df)}")

# Create class mapping from the 'common_name' column
class_names = sorted(ena24_df['common_name'].unique())
class_map = {name: i for i, name in enumerate(class_names)}

# Define the output directory for labels and save the class names
labels_base_dir = os.path.join(base_data_path_yolo)
os.makedirs(labels_base_dir, exist_ok=True)
with open(os.path.join(labels_base_dir, 'classes.txt'), 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")
print(f"Saved {len(class_names)} class names to {os.path.join(labels_base_dir, 'classes.txt')}")


# Take a sample of images for demonstration
NUM_IMAGES_LABEL=500
sample_images = ena24_df.sample(NUM_IMAGES_LABEL, random_state=42) # Use a random state for reproducibility

# Initialize accuracy tracker
accuracy_tracker = {name: {'detected': 0, 'missed': 0, 'total': 0} for name in sorted(sample_images['common_name'].unique())}

# Use a general prompt for object detection
texts = [["a photo of an animal", "a photo of a bird", "a photo of a dog"]]

index_img = 0
for index, row in tqdm.tqdm(sample_images.iterrows()):
    image_relative_path = row['filepath']
    full_image_path = os.path.join(base_data_path, 'data/test/', image_relative_path)
    common_name = row['common_name']
    
    if os.path.exists(full_image_path):
        try:
            # print(f"Processing image: {full_image_path}")
            image = Image.open(full_image_path).convert("RGB")
            
            accuracy_tracker[common_name]['total'] += 1
            
            # Prepare inputs for OWL2 model
            inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Post-process the outputs
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)

            i = 0  # Predictions for the first (and only) image
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            # If any objects are detected, process the highest-confidence one
            if len(scores) > 0:
                accuracy_tracker[common_name]['detected'] += 1
                
                # Find the detection with the highest score
                best_score_index = scores.argmax()
                best_box = boxes[best_score_index]
                
                # Get the ground truth class ID from the CSV
                class_id = class_map[common_name]

                # Convert bounding box to YOLO format (normalized)
                img_width, img_height = image.size
                x_min, y_min, x_max, y_max = best_box.tolist()
                
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                box_width = x_max - x_min
                box_height = y_max - y_min

                norm_x_center = x_center / img_width
                norm_y_center = y_center / img_height
                norm_width = box_width / img_width
                norm_height = box_height / img_height

                # Define the path for the YOLO label file
                label_relative_path = os.path.splitext(image_relative_path)[0] + '.txt'
                label_full_path = os.path.join(labels_base_dir, label_relative_path)
                os.makedirs(os.path.dirname(label_full_path), exist_ok=True)

                # Write the YOLO label file
                with open(label_full_path, 'w') as f:
                    f.write(f"{class_id} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
                # print(f"Saved YOLO label for '{common_name}' to {label_full_path}")

                # Define paths for YOLO training data
                yolo_train_images_dir = os.path.join(labels_base_dir, 'images')
                yolo_train_labels_dir = os.path.join(labels_base_dir, 'labels')
                os.makedirs(yolo_train_images_dir, exist_ok=True)
                os.makedirs(yolo_train_labels_dir, exist_ok=True)

                # Copy image to YOLO training images directory
                image_name = os.path.basename(full_image_path)
                destination_image_path = os.path.join(yolo_train_images_dir, image_name)
                shutil.copyfile(full_image_path, destination_image_path)
                # print(f"Copied image to {destination_image_path}")

                # Copy label file to YOLO training labels directory
                label_name = os.path.basename(label_full_path)
                destination_label_path = os.path.join(yolo_train_labels_dir, label_name)
                shutil.copyfile(label_full_path, destination_label_path)
                # print(f"Copied label to {destination_label_path}")
            else:
                accuracy_tracker[common_name]['missed'] += 1
                # print(f"No animal detected in image for '{common_name}'")

            # Visualize the detections on the image for verification
            if index_img < 10:
                image_with_boxes = image.copy()
                draw = ImageDraw.Draw(image_with_boxes)

                for box, score, label in zip(boxes, scores, labels):
                    box = [round(i, 2) for i in box.tolist()]
                    detected_text = texts[0][label.item()]
                    # print(
                    #     f"Detected '{detected_text}' with confidence {round(score.item(), 3)} at location {box}"
                    # )
                    draw.rectangle(box, outline="red", width=3)
                    draw.text((box[0], box[1]), f"{detected_text} {round(score.item(), 3)}", fill="red")

                display(image_with_boxes)

        except Exception as e:
            print(f"Could not process image {full_image_path}: {e}")
    else:
        print(f"Image file not found: {full_image_path}")
    index_img += 1
```

## Automatic Labelling Accuracy
```python
# After the loop, print the accuracy summary
print("\n--- OWL2 Detection Accuracy Summary ---")
for common_name, stats in accuracy_tracker.items():
    total = stats['total']
    if total > 0:
        detected_fraction = stats['detected'] / total
        missed_fraction = stats['missed'] / total
        print(f"Class: {common_name}, total: {total}, detected: {stats['detected']} ({detected_fraction:.2%}), missed: {stats['missed']} ({missed_fraction:.2%})")
print("-------------------------------------\n")
```