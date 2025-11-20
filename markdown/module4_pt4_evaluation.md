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
# Path to the directory where all YOLO training runs are saved
train_dir = 'runs/detect'

# Find the most recent training directory by sorting them by modification time
latest_train_run = max(os.listdir(train_dir), key=lambda d: os.path.getmtime(os.path.join(train_dir, d)))
# The best model weights are saved as 'best.pt' inside the 'weights' subdirectory
best_model_path = os.path.join(train_dir, latest_train_run, 'weights/best.pt')

print(f"Loading fine-tuned model from: {best_model_path}")

# Load the fine-tuned model from the best weights file
model_finetuned = YOLO(best_model_path)

# Get the paths of up to 9 validation images to display
if len(val_images) < 9:
    print(f"Warning: Found only {len(val_images)} validation images. Displaying all of them.")
    display_images = val_images
else:
    # Select the first 9 images from the validation set
    display_images = val_images[:9]

if not display_images:
    raise Exception("No validation images found to display results.")

# Create a 3x3 grid for displaying the images and their predictions
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten() # Flatten the 2D array of axes into a 1D array for easy iteration

# Run inference on the selected images and display the results
for i, image_path in enumerate(display_images):
    # Note: YOLO expects the original image path, not the pre-processed one
    print(f"Running inference on: {image_path}")
    # Run the fine-tuned model on the image
    results = model_finetuned(image_path)
    
    # The `plot()` method returns a numpy array of the image with bounding boxes and labels drawn on it
    im_array = results[0].plot()
    # Convert the array from BGR (used by OpenCV) to RGB for correct display with PIL and Matplotlib
    im = Image.fromarray(im_array[..., ::-1])
    
    # Display the image in the current subplot
    axs[i].imshow(im)
    axs[i].axis('off') # Hide the x and y axes
    axs[i].set_title(os.path.basename(image_path))

# Hide any unused subplots if there are fewer than 9 images
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

# Run the `val` method to evaluate the model on the validation set.
# This uses the 'val' dataset defined in the 'ena24_yolo_dataset.yaml' file.
# The method calculates metrics like mAP, precision, and recall, and saves results to a new run directory.
metrics = model_finetuned.val()

# The validation process automatically generates and saves a confusion matrix image.
# We can find its path in the `save_dir` attribute of the returned metrics object.
confusion_matrix_path = os.path.join(metrics.save_dir, 'confusion_matrix.png')

# Check if the confusion matrix image was created successfully
if os.path.exists(confusion_matrix_path):
    print(f"Displaying confusion matrix from: {confusion_matrix_path}")
    # Open and display the confusion matrix image
    img = Image.open(confusion_matrix_path)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off') # Hide the axes for a cleaner look
    plt.title('Confusion Matrix')
    plt.show()
else:
    print(f"Confusion matrix not found at: {confusion_matrix_path}")
```
