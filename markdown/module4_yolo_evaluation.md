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
# YOLO Fine-Tuned Model Evaluation

This notebook evaluates a fine-tuned YOLO model. It assumes that you have already run the `module4_yolo_finetuning.md` notebook to train the model.

This notebook will:
1. Load the best model from the latest training run.
2. Display predictions on a sample of validation images.
3. Visualize the training and validation loss curves.
4. Evaluate the model on the entire validation set and display the confusion matrix.

First, let's import the required libraries.
<!-- #endregion -->

```python
import os
import yaml
from ultralytics import YOLO
from PIL import Image
from IPython.display import display
import pandas as pd
import numpy as np
import seaborn as sns
import tqdm

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
```

<!-- #region -->
## Setup Paths

We need to define the paths to the dataset and validation files. This should be consistent with the `module4_yolo_finetuning.md` notebook.
<!-- #endregion -->

```python
base_path = os.path.abspath('../data/IDLE-OO-Camera-Traps_yolo')
val_file_path = os.path.join(base_path, 'val.txt')

with open(val_file_path, 'r') as f:
    val_images = [line.strip() for line in f.readlines()]
print(f"Found {len(val_images)} validation images.")
```

<!-- #region -->
## View Results

After training, the best model is saved in the `runs/detect/train/weights/` directory. Let's load this model and run inference on one of the images used for training to verify that the model has learned.

We will select one of the 10 sample images that were labeled.

<!-- #endregion -->

```python
# Path to the directory where training runs are saved
train_dir = 'runs/detect'

# Find the latest training directory
latest_train_run = max(os.listdir(train_dir), key=lambda d: os.path.getmtime(os.path.join(train_dir, d)))
best_model_path = os.path.join(train_dir, latest_train_run, 'weights/best.pt')

print(f"Loading fine-tuned model from: {best_model_path}")

# Load the fine-tuned model
model_finetuned = YOLO(best_model_path)

# Get the paths of up to 9 validation images
if len(val_images) < 9:
    print(f"Warning: Found only {len(val_images)} validation images. Displaying all of them.")
    display_images = val_images
else:
    display_images = val_images[:9]

if not display_images:
    raise Exception("No validation images found to display results.")

# Create a 3x3 grid for displaying the images
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten()

# Run inference and display results
for i, image_path_relative in enumerate(display_images):
    image_path = os.path.join(base_path, image_path_relative)
    print(f"Running inference on: {image_path}")
    results = model_finetuned(image_path)
    
    # Plot results and convert to an image
    im_array = results[0].plot()
    im = Image.fromarray(im_array[..., ::-1])
    
    # Display the image in the subplot
    axs[i].imshow(im)
    axs[i].axis('off') # Hide axes
    axs[i].set_title(os.path.basename(image_path))

# Hide any unused subplots
for j in range(i + 1, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.show()
```

<!-- #region -->
## Evaluate on Validation Set and Visualize Confusion Matrix

Now that we have a fine-tuned model, let's evaluate its performance on the entire validation set. This will give us metrics like mAP (mean Average Precision) and also allow us to generate a confusion matrix to see how well the model distinguishes between different classes.

The `val()` method will run prediction on all images in the validation set defined in `ena24_yolo_dataset.yaml`.
<!-- #endregion -->

```python
# Path to the directory where training runs are saved
train_dir = 'runs/detect'

# Find the latest training directory
latest_train_run = max(os.listdir(train_dir), key=lambda d: os.path.getmtime(os.path.join(train_dir, d)))
best_model_path = os.path.join(train_dir, latest_train_run, 'weights/best.pt')

print(f"Loading fine-tuned model from: {best_model_path}")

# Load the fine-tuned model
model_finetuned = YOLO(best_model_path)

# Run validation on the full validation set
metrics = model_finetuned.val()

# The confusion matrix is saved by the val command. Let's display it.
confusion_matrix_path = os.path.join(metrics.save_dir, 'confusion_matrix.png')

# Check if the confusion matrix image exists
if os.path.exists(confusion_matrix_path):
    print(f"Displaying confusion matrix from: {confusion_matrix_path}")
    # Display the confusion matrix
    img = Image.open(confusion_matrix_path)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Confusion Matrix')
    plt.show()
else:
    print(f"Confusion matrix not found at: {confusion_matrix_path}")
```
