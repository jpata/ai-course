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

```python
#Mount the drive in colab to be able to share outputs across the notebooks
import sys
import os
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive/')

    %mkdir -p /content/drive/MyDrive/ai-course
    %cd /content/drive/MyDrive/ai-course

    if not os.path.exists('ai-course'):
        !git clone https://github.com/jpata/ai-course
    
    %cd ai-course
    !git pull
```

<!-- #region -->
# Object Detection with OWL2

This notebook demonstrates how to use the OWL2 model for object detection on the IDLE-OO-Camera-Traps dataset.

First, let's install the necessary libraries. We need `transformers` for the OWL2 model, `datasets` to handle the data, `torch` and `torchvision` as the backend for the model, `Pillow` for image manipulation, `ultralytics` for potential YOLO format compatibility, and `scikit-learn` for utilities like PCA.

```python
!pip install -q transformers datasets torch torchvision Pillow ultralytics scikit-learn
```

Now, let's import the required libraries. These include `torch` for tensor operations, components from `transformers` and `datasets` for the model and data, `PIL` (Pillow) for image processing, and `matplotlib` for plotting.
<!-- #endregion -->

```python
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
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

We will load the OWL2 model and processor from Hugging Face. The `Owlv2Processor` is responsible for preparing the inputs for the model (both image and text), and the `Owlv2ForObjectDetection` is the model itself. We are using the `google/owlv2-base-patch16-ensemble` checkpoint.

```python
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
```

Now, let's define the objects we want to detect. We can see a cheetah in the image, so let's try to detect that. The model is zero-shot, so we can provide arbitrary text queries. Here, we are providing a few example queries. Note that these queries are not ideal for detecting a cheetah, but they will demonstrate the model's ability to distinguish between different objects. The processor then tokenizes the text and preprocesses the image to create the model inputs.

```python
texts = [["a photo of a leopard", "a photo of a tiger", "a photo of a rock"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
```

Now we run the model to get the object detection outputs. We use `torch.no_grad()` as we are doing inference and don't need to compute gradients. The model returns a dictionary of outputs, including logits and predicted boxes.

```python
with torch.no_grad():
  outputs = model(**inputs)
```


The model outputs logits and bounding boxes in a raw format. We use the `processor.post_process_object_detection` function to convert these into human-readable predictions. This function filters detections based on a `threshold`, and rescales the bounding boxes to the original image size. We then iterate through the detections and draw them on the image.

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

This code block dives into the model's internal workings.
1.  `image_embedder`: We first pass the image through the `image_embedder` (the Vision Transformer backbone) to get a `feature_map`. This map represents the image as a grid of feature vectors, one for each image patch.
2.  `class_predictor`: We get `source_class_embeddings` which are class-agnostic embeddings for each patch.
3.  `objectness_predictor`: This gives a score for each patch indicating how likely it is to contain any object.
4.  `box_predictor`: This predicts the bounding box for each patch.
5.  **PCA Visualization**: To visualize the rich information in the `source_class_embeddings`, we use Principal Component Analysis (PCA) to reduce their dimensionality from 768 to 3. This allows us to view the embeddings as an RGB image, where different colors represent different semantic features detected in the patches.

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

Here we visualize the PCA-reduced class embeddings. The resulting image shows how the model groups different parts of the image semantically. Areas with similar colors are considered similar by the model. You can often see the main object (the cheetah) being segmented from the background.

```python
plt.title("Class embeddings")
plt.imshow(pca_image)
```

This plot shows the objectness scores for each patch. Brighter areas indicate a higher probability that the patch contains an object. Notice how the model highlights the area where the cheetah is located.

```python
plt.title("Objectness scores")
plt.imshow(objectnesses.detach().numpy().reshape(H,W))
```

This is a helper function to reverse the preprocessing steps applied by the `Owlv2Processor`. The processor normalizes the image (e.g., by subtracting the mean and dividing by the standard deviation of the training data). This function "un-normalizes" the image so we can visualize it correctly with `matplotlib`.

```python
def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(processor.image_processor.image_std)[:, None, None]) + np.array(processor.image_processor.image_mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image
```

Here, we identify the top 3 patches with the highest objectness scores. We then take the raw bounding box predictions from the `box_predictor` for these patches and draw them on the "un-normalized" image. This gives us a glimpse into the model's raw output before any post-processing like non-maximum suppression. The boxes are predicted as `(center_x, center_y, width, height)` in a normalized format, so we need to scale them to the image size.

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

    # Get the objectness score for this patch
    objectness_score = objectnesses.detach().numpy()[0, patch_idx]

    # Draw the rectangle
    draw.rectangle(box_coords_pixel, outline="lime", width=3)
    # Draw the objectness score
    draw.text((box_coords_pixel[0], box_coords_pixel[1]), f"Confidence: {objectness_score:.3f}", fill="lime")

plt.imshow(image_with_raw_boxes)
```

## Image guided prompting

Instead of text, we can also use an image or part of an image as a prompt. This is called image-guided or one-shot object detection.

First, let's get a few more images from the dataset to search in.

```python
iterator = iter(dataset)
target_images = [next(iterator)["image"] for i in range(5)]
```

Now, we'll take the class embedding of the object we detected with the highest confidence in the previous steps (which happened to be a cheetah-like figure in the rock). This embedding will serve as our "query". We then loop through the new `target_images`. For each target image, we compute its class embeddings and compare them with our query embedding. The `class_predictor` can take a query embedding to condition its output. We then find the patch in the target image whose embedding is most similar to our query embedding and draw its bounding box. This allows us to find "more objects like this one".

```python
from scipy.special import expit as sigmoid
query_embedding = source_class_embeddings[0][top_scores[-1]]
for target_image in target_images:
    target_pixel_values = processor(images=target_image, return_tensors="pt").pixel_values
    unnormalized_target_image = get_preprocessed_image(target_pixel_values)
    
    with torch.no_grad():
      feature_map = model.image_embedder(target_pixel_values)[0]
    
    # Get boxes and class embeddings (the latter conditioned on query embedding)
    b, h, w, d = feature_map.shape
    target_boxes = model.box_predictor(
        feature_map.reshape(b, h * w, d), feature_map=feature_map
    )
    
    target_class_predictions = model.class_predictor(
        feature_map.reshape(b, h * w, d),
        torch.tensor(query_embedding[None, None, ...]),  # [batch, queries, d]
    )[0]
    
    # Remove batch dimension and convert to numpy:
    target_boxes = np.array(target_boxes[0].detach())
    target_logits = np.array(target_class_predictions[0].detach())
    
    # Take the highest scoring logit
    top_ind = np.argmax(target_logits[:, 0], axis=0)
    score = sigmoid(target_logits[top_ind, 0])
    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(unnormalized_target_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    
    # Get the corresponding bounding box
    cx, cy, w, h = target_boxes[top_ind]
    ax.plot(
        [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
        [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
        color='lime',
    )
    
    ax.text(
        cx - w / 2 + 0.015,
        cy + h / 2 - 0.015,
        f'Score: {score:1.2f}',
        ha='left',
        va='bottom',
        color='lime',
        # bbox={
        #     #'facecolor': 'white',
        #     'edgecolor': 'lime',
        #     'boxstyle': 'square,pad=.3',
        # },
    )
    
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title(f'Closest match')
```

## Automatic data labelling

One powerful application of zero-shot object detectors like OWL2 is automatic data labeling. We can use the model to generate initial labels for a dataset, which can then be manually reviewed and corrected, saving a lot of time.

This code block demonstrates this process:
1.  **Setup**: We set the device to GPU if available, define paths, and load a CSV file (`ENA24-balanced.csv`) that contains file paths and the true `common_name` for images in the ENA24 dataset.
2.  **Class Mapping**: We create a mapping from the string `common_name` to an integer `class_id`, which is standard for training object detection models. We save these class names to `classes.txt`.
3.  **Image Sampling**: We take a sample of images to label.
4.  **Loop and Detect**: We loop through each sampled image.
5.  **Run OWL2**: For each image, we run OWL2 with general prompts like "a photo of an animal".
6.  **Process Detections**: If the model detects an object, we take the one with the highest confidence score.
7.  **Create YOLO Label**: We convert the predicted bounding box into the YOLO format (`<class_id> <x_center> <y_center> <width> <height>`, all normalized).
8.  **Save Files**: We save the YOLO label to a `.txt` file and copy both the image and the label file into a new directory structure (`images/` and `labels/`) suitable for training a YOLO model.
9.  **Track Accuracy**: We keep track of how many images for each class had at least one object detected versus how many were missed. This gives us a rough idea of OWL2's performance on this dataset with our general prompts.
10. **Visualize**: For the first few images, we display the image with the predicted bounding boxes for visual inspection.

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
NUM_IMAGES_LABEL=1000
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

## Automatic Labelling Detection Rate

Finally, we print the summary of our quick analysis. This shows, for each animal class, how many images we processed, and what percentage of them had at least one object detected by OWL2. This is not a measure of classification accuracy, but rather "detection recall" with a very general prompt. It helps to understand which animals are more easily detected by the model out-of-the-box.
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

## Challenge: How Would You Measure Detection Precision?

The "Automatic Labelling Detection Rate" we calculated gives us a sense of **Recall** in a specific way: out of all the images that *truly contain* an animal (based on our dataset's `common_name` label), what percentage of them did our model successfully detect *at least one object*? This metric tells us how good the model is at finding *something* when an animal is present.

However, it doesn't tell us about the model's **Precision** regarding the detections themselves. In this context, precision would address: of all the bounding boxes the model *generated*, how many were genuinely identifying the animal, rather than some other object or a spurious detection?

For example, our current script might flag an image as "detected" if OWL2 puts a bounding box around a tree, a rock, or some other background element, even though the image is known to contain an animal. This would be a **False Positive** detection. The model's task here is not to identify the animal's species, but simply to localize the animal.

**Your challenge:** How would you modify the automatic labeling script to measure this form of detection precision? Think about:
1.  Currently, the `texts` prompts are quite general (e.g., `"a photo of an animal"`, `"a photo of a bird"`, `"a photo of a dog"`). How could you use the `label` returned by OWL2 (which corresponds to one of these prompts) to infer if the detection is likely a false positive?
2.  If the image is known to contain an animal, and the model's highest-scoring detection corresponds to a prompt like `"a photo of a rock"`, how would you count that?
3.  What counters would you need to track (e.g., `true_positive_detections`, `false_positive_detections`) to calculate precision for localization?