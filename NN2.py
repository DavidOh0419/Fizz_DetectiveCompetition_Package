# -*- coding: utf-8 -*-

import os
import math
import re
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers

from sklearn.utils import shuffle
import shutil

# -----------------------------
# Paths (relative to this file)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(BASE_DIR, "samples")      # input samples
save_path = os.path.join(BASE_DIR, "gen_samples")    # generated / labeled chars

os.makedirs(save_path, exist_ok=True)

# -----------------------------
# Helper: clear generated folder
# -----------------------------
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)           # delete file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)       # delete subfolder
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

clear_folder(save_path)

# -----------------------------
# Mapping for generated samples
# -----------------------------
# letter -> running index for saved images
char_counter = {}

for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
    char_counter[ch] = 0

# -----------------------------
# Data augmentation & slicing
# -----------------------------
all_files = os.listdir(folder_path)
full_paths = [os.path.join(folder_path, f) for f in all_files]

for file_name in full_paths:
    if not os.path.isfile(file_name):
        continue

    NUM_IMAGES_TO_GENERATE = 8

    plate_name = os.path.splitext(os.path.basename(file_name))[0]
    parts = plate_name.split('_')
    if len(parts) < 2:
        print(f"Skipping {file_name}: filename doesn't contain label part")
        continue

    letters = list(parts[1])

    # Load image as NumPy array
    pil_img = Image.open(file_name).convert('RGB')
    base_img = np.array(pil_img)
    h0, w0, _ = base_img.shape

    if w0 < 12:
        print(f"Skipping {file_name}: width {w0} < 12, cannot split into 12 sections")
        continue

    image_array = np.expand_dims(base_img, 0)

    datagen = ImageDataGenerator(
        rotation_range=1, 
        zoom_range=0.001,
        brightness_range=[0.2, 1.5],
        shear_range=6
    )

    datagen_iterator = datagen.flow(image_array, batch_size=1)

    for _ in range(NUM_IMAGES_TO_GENERATE):
        value = next(datagen_iterator)
        img = value[0].astype('uint8')
        h, w, _ = img.shape
        mid = h // 2

        # upper/lower split of plate
        img_upper = img[50:mid-50, :]
        img_lower = img[mid+50:h-50, :]

        # crop left/right margins
        if w > 60:
            img_lower = img_lower[:, 30:w-30]

        # Compute section width for 12 characters
        section_width = (img_lower.shape[1] // 12)
        if section_width <= 0:
            print(f"Warning: augmented image from {file_name} has too small width {img_lower.shape[1]} — skipping this sample")
            continue

        sections = []
        for i in range(12):
            x0 = i * section_width
            x1 = (i + 1) * section_width
            section_img = img_lower[:, x0:x1]

            if section_img.size == 0:
                print(f"Empty section at index {i} for file {file_name} (x0={x0}, x1={x1}), skipping")
                continue

            sections.append(section_img)

        for i, section_img in enumerate(sections):
            if i >= len(letters):
                break  # not enough label chars

            letter = letters[i]
            save_filename = f"{letter}_{char_counter[letter]}.png"
            char_counter[letter] += 1
            path = os.path.join(save_path, save_filename)

            # *** CHANGED: convert to grayscale & keep consistent size later ***
            # For now we save as-is; resizing happens after loading.
            cv2.imwrite(path, section_img)

# -----------------------------
# Helper: list files in folder
# -----------------------------
def files_in_folder(folder_path):
    """
    Returns a sorted list of file names in folder_path.
    """
    try:
        files_A = os.listdir(folder_path)
    except Exception as e:
        print("Error reading folder:", e)
        return []

    files_A.sort()
    return files_A

# -----------------------------
# Build X (images) and Y (labels)
# -----------------------------
PATH = save_path

X = []
Y = []

labeled_set = files_in_folder(PATH)

for file_name in labeled_set:
    img_path = os.path.join(PATH, file_name)
    img = Image.open(img_path).convert("L")  # grayscale
    img = np.array(img)

    # *** OPTIONAL: ensure a fixed size (uncomment & choose a size) ***
    # img = cv2.resize(img, (45, 200))  # (width, height)

    parts = file_name.split("_")
    label_char = parts[0]  # e.g. "A" from "A_0.png"

    X.append(img)
    Y.append(label_char)

# Shuffle data
X, Y = shuffle(X, Y, random_state=42)

X = np.array(X)   # shape: (N, H, W)
Y = np.array(Y)

print("Raw X.shape:", X.shape)  # sanity check, e.g. (N, 200, 45)

# -----------------------------
# Map labels to integers
# -----------------------------
label_mapping = {}
for i, ch in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    label_mapping[ch] = i

Y = np.array([label_mapping[ch] for ch in Y])

# -----------------------------
# One-hot encoding
# -----------------------------
NUMBER_OF_LABELS = 36
CONFIDENCE_THRESHOLD = 0.01   # (not really used below but kept)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

# -----------------------------
# Normalize X (images) dataset
# -----------------------------
# *** CHANGED: infer H, W from data and match the model to it ***
N, H, W = X.shape
print(f"Inferred image size: H={H}, W={W}")

X_dataset = X.reshape(-1, H, W, 1).astype("float32") / 255.0
Y_dataset = convert_to_one_hot(Y, NUMBER_OF_LABELS)

print("Y_dataset shape:", Y_dataset.shape)

# -----------------------------
# Train/validation split
# -----------------------------
VALIDATION_SPLIT = 0.2

print(
    "Total examples: {:d}\nTraining examples: {:d}\nTest examples: {:d}".format(
        X_dataset.shape[0],
        math.ceil(X_dataset.shape[0] * (1 - VALIDATION_SPLIT)),
        math.floor(X_dataset.shape[0] * VALIDATION_SPLIT),
    )
)
print("X shape:", X_dataset.shape)
print("Y shape:", Y_dataset.shape)

split_index = math.ceil(X_dataset.shape[0] * (1 - VALIDATION_SPLIT))

X_train_dataset = X_dataset[:split_index]
Y_train_dataset = Y_dataset[:split_index]
X_val_dataset   = X_dataset[split_index:]
Y_val_dataset   = Y_dataset[split_index:]

# -----------------------------
# Model definition
# -----------------------------
LEARNING_RATE = 1e-4

conv_model = models.Sequential()
conv_model.add(
    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same",          # *** CHANGED: padding="same" to control shrinkage ***
        input_shape=(H, W, 1)    # *** CHANGED: dynamic input shape ***
    )
)
conv_model.add(layers.MaxPooling2D((2, 2)))  # -> (H/2, W/2)

conv_model.add(
    layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    )
)
conv_model.add(layers.MaxPooling2D((2, 2)))  # -> (H/4, W/4)

conv_model.add(
    layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    )
)
conv_model.add(layers.MaxPooling2D((2, 2)))  # -> (H/8, W/8)

conv_model.add(
    layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    )
)
# *** CHANGED: still safe 2x2 pool because we used padding="same"
conv_model.add(layers.MaxPooling2D((2, 2)))  # -> (H/16, W/16) but stays >=1

conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation="relu"))
conv_model.add(layers.Dense(NUMBER_OF_LABELS, activation="softmax"))

conv_model.summary()

conv_model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE),
    metrics=["acc"],
)

# -----------------------------
# Train the model
# -----------------------------
history_conv = conv_model.fit(
    X_train_dataset,
    Y_train_dataset,
    validation_data=(X_val_dataset, Y_val_dataset),
    epochs=80,
    batch_size=16,
)

# -----------------------------
# Save model
# -----------------------------
conv_model.save(os.path.join(BASE_DIR, "clue_char_model.h5"))
conv_model.save(os.path.join(BASE_DIR, "clue_char_model.keras"))

# -----------------------------
# Plot training curves
# -----------------------------
plt.figure()
plt.plot(history_conv.history["loss"])
plt.plot(history_conv.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train loss", "val loss"], loc="upper left")
plt.show()

plt.figure()
plt.plot(history_conv.history["acc"])
plt.plot(history_conv.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train accuracy", "val accuracy"], loc="upper left")
plt.show()

# -----------------------------
# Simple test display
# -----------------------------
def display_image(index):
    img = X_dataset[index]
    img_aug = np.expand_dims(img, axis=0)
    y_predict = conv_model.predict(img_aug)[0]

    plt.imshow(img.squeeze(), cmap="gray")
    caption = "Predicted class: {} (conf {:.2f})".format(
        np.argmax(y_predict),
        float(y_predict[np.argmax(y_predict)]),
    )
    plt.title(caption)
    plt.axis("off")
    plt.show()

# Example: show first image and prediction
display_image(0)
