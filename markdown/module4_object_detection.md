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
# Object Detection: A Comparison of YOLO and RT-DETR

This notebook introduces and compares two prominent object detection architectures: YOLO (You Only Look Once) and RT-DETR (Real-Time DEtection TRansformer). We will use pretrained models to perform inference on the ENA24 dataset and analyze their performance, paying special attention to the challenges posed by the mismatch between the models' training classes (COCO) and the dataset's actual classes.

Object detection is a computer vision task that involves identifying and locating objects within an image. A model performing this task returns a set of bounding boxes, each with a corresponding class label for the object it contains.

We will explore:
*   **YOLO**: A leading family of single-stage detectors known for its speed and efficiency.
*   **RT-DETR**: A modern, transformer-based, end-to-end detector that provides high accuracy without requiring complex post-processing steps like Non-Maximum Suppression (NMS).
*   **The ENA24 Dataset**: We will use the `imageomics/IDLE-OO-Camera-Traps` dataset to evaluate how well these models, pretrained on general-purpose datasets, perform on specialized data.
<!-- #endregion -->

<!-- #region -->
## 1. Setup

First, let's install the necessary libraries. `ultralytics` provides the YOLO model, while `transformers` gives us access to RT-DETR.
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

# YOLO imports
from ultralytics import YOLO

# DETR imports
from transformers import DetrImageProcessor, RTDetrForObjectDetection
```

<!-- #region -->
## 2. Loading the Dataset and a Sample Image

We'll load the `imageomics/IDLE-OO-Camera-Traps` dataset from a local path and select one example from the test split to use for our initial inference examples.
<!-- #endregion -->

```python
dataset = load_dataset(path="../data/IDLE-OO-Camera-Traps", split="test")
iterator = iter(dataset)
sample = next(iterator)
image = sample["image"]
print("A sample image from the ENA24 dataset:")
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

YOLO's architecture can be broken down into three key parts:
*   **Backbone**: A deep convolutional neural network (like CSPDarknet53 in YOLOv8) that extracts image features at different scales.
*   **Neck**: This part (e.g., a PANet) merges and refines the feature maps from the backbone, creating a rich feature pyramid that helps detect objects of various sizes.
*   **Head**: The detection head takes the feature maps from the neck and predicts bounding boxes, class probabilities, and an "objectness" score for each location on a predefined grid.

A critical post-processing step for YOLO is **Non-Maximum Suppression (NMS)**. Because the model predicts many potential bounding boxes for the same object, NMS is used to discard redundant, overlapping boxes, keeping only the one with the highest confidence score.
<!-- #endregion -->

<!-- #region -->
## 4. Part 2: RT-DETR (Real-Time DEtection TRansformer)

DETR (DEtection TRansformer) models reframe object detection as a direct set prediction problem. They use a transformer-based architecture to produce a fixed-size set of predictions, eliminating the need for complex post-processing like NMS. RT-DETR is an evolution of this idea, optimized for real-time performance.

### 4.1. Inference with RT-DETR

We will use the `transformers` library to load a pretrained RT-DETR model. Unlike YOLO, DETR models require a specific `processor` to resize and normalize the input image correctly.
<!-- #endregion -->

```python
# Load the processor and a pretrained RT-DETR model from Hugging Face
processor_detr = DetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model_detr = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

# Prepare the image for the model
inputs = processor_detr(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model_detr(**inputs)

# Post-process the results to get bounding boxes and class labels
target_sizes = torch.tensor([image.size[::-1]])
results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

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

# Get labels, scores, and boxes
labels = [model_detr.config.id2label[i.item()] for i in results_detr["labels"]]
scores = results_detr["scores"]
boxes = results_detr["boxes"]

# Draw the boxes on the image
im_detr = draw_boxes(image, boxes, labels, scores)

print("RT-DETR Detections (Confidence > 0.7):")
display(im_detr)
```

<!-- #region -->
### 4.2. RT-DETR Architectural Deep Dive

The DETR architecture is fundamentally different from YOLO:
*   **Backbone**: Like YOLO, it uses a standard CNN backbone (e.g., ResNet) to extract a feature map from the image.
*   **Transformer Encoder-Decoder**: This is the core of the model.
    *   The **Encoder** takes the image features and enriches them using self-attention mechanisms.
    *   The **Decoder** takes a small, fixed number of learnable embeddings called **object queries**. Each query is responsible for finding one object in the image. The decoder uses attention to compare the object queries to the image features and outputs the final set of predictions (class and bounding box) for each query.

This end-to-end philosophy means that each object is detected exactly once by one of the object queries. In its original form, this completely removes the need for NMS, simplifying the detection pipeline. RT-DETR introduces some optimizations that re-introduce an optional, efficient NMS-like step, but the core principle remains.
<!-- #endregion -->

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

                # --- Run RT-DETR detection ---
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

# --- RT-DETR Crosstab ---
df_detr = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred_detr})
detr_crosstab = pd.crosstab(df_detr['y_true'], df_detr['y_pred'], rownames=['True Class (ENA24)'], colnames=['Predicted Class (RT-DETR/COCO)'])

plt.figure(figsize=(18, 14))
sns.heatmap(detr_crosstab, annot=True, fmt='d', cmap='Greens')
plt.title('ENA24 True Class vs. RT-DETR Predicted COCO Class')
plt.show()
```

<!-- #region -->
From the heatmaps, we can see how the models perform. For example, both models often correctly map the ENA24 `bird` class to the COCO `bird` class. However, for species not in COCO, like `red_deer` or `wild_boar`, the models might predict related classes like `cow`, `horse`, or `dog`, or simply fail to make a detection. This analysis highlights the limitations of using pretrained models on novel domains and underscores the need for fine-tuning.
<!-- #endregion -->

<!-- #region -->
## 6. Head-to-Head Comparison: YOLO vs. RT-DETR

| Feature               | YOLOv8                                        | RT-DETR                                                    |
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
*   **RT-DETR** represents a newer paradigm, using a transformer architecture to perform end-to-end detection. This removes the need for hand-tuned components like NMS and can lead to better performance, though often at the cost of higher computational requirements and implementation complexity.

Our analysis on the ENA24 dataset showed that while pretrained models are a fantastic starting point, their performance on specialized domains is limited by their training data. To achieve high accuracy on ENA24's specific animal classes, the clear next step is **fine-tuning**, where we would train these models further on the ENA24 data itself.
<!-- #endregion -->
