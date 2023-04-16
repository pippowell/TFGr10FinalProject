import tensorflow_datasets as tfds
import tensorflow as tf
import splitfolders
from pathlib import Path
from model import phi_values
import cv2
import os
import numpy as np

version = "b0"
_, _, res, _ = phi_values[version]
batch_size = 1 # since the dataset is small

# Load and preprocess first dataset
dataset1_folder = str(Path(__file__).parents[0]) + '/images/bananas'  # Replace with the path to your first dataset folder
images1 = []
labels1 = []

for file_name in os.listdir(dataset1_folder):
    file_path = os.path.join(dataset1_folder, file_name)
    # Load the image using OpenCV
    image = cv2.imread(file_path)
    # Resize the image to a specific size
    image = cv2.resize(image, (res, res))
    # Normalize the pixel values to [0, 1]
    image = image/255.0
    images1.append(image)
    label = 0 # Replace with the label for dataset1 images
    labels1.append(label)

# Load and preprocess second dataset
dataset2_folder = str(Path(__file__).parents[0]) + '/images/grapes' # Replace with the path to your second dataset folder
images2 = []
labels2 = []
for file_name in os.listdir(dataset2_folder):
    file_path = os.path.join(dataset2_folder, file_name)
    # Load the image using OpenCV
    image = cv2.imread(file_path)
    # Resize the image to a specific size
    image = cv2.resize(image, (res, res))
    # Normalize the pixel values to [0, 1]
    image = image/255.0
    images2.append(image)
    label = 1  # Replace with the label for dataset2 images
    labels2.append(label)

# Concatenate or merge the datasets
images = np.concatenate((images1, images2), axis=0)
labels = np.concatenate((labels1, labels2), axis=0)

images = np.array(images)
labels = np.array(labels)

data = tf.data.Dataset.from_tensor_slices((images, labels))

# Shuffle the order of images
data = data.shuffle(1000)

# Split into training and validation sets
train_ds = data.take(int(len(images)*0.8))  # 80% of data for training
val_ds = data.skip(int(len(images)*0.8))  # 20% of data for validation

# Further processing and batching of the datasets
train_ds = train_ds.batch(batch_size) 
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.batch(batch_size)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# print(f"train_ds: {train_ds.cardinality().numpy()}")
# print(f"val_ds: {val_ds.cardinality().numpy()}")
