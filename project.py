import os
import cv2
import numpy as np
import tensorflow as tf
import gdown
from collections import Counter
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import math

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1SXUiI7bYK-6CaLPof6CaB2Eh57B07pJn"
MODEL_PATH = "SkinLesionAI.h5"
PATIENT_DIR = os.path.join(os.getcwd(), 'patient_images')
IMG_SIZE = (224, 224)
mapping = {0: 'benign', 1: 'malignant'}

# Initialize tkinter root for pop-up dialogs
root = tk.Tk()
root.withdraw()  # Hide the main window

# Function to show information pop-up
def show_info_message(message):
    messagebox.showinfo("Info", message)

# Function to show error pop-up
def show_error_message(message):
    messagebox.showerror("Error", message)

# Function to download the model weights if not present
def download_model_weights(model_path, model_url):
    if not os.path.exists(model_path):
        show_info_message("Downloading the model weights 'SkinLesionAI.h5' from Google Drive...")
        try:
            gdown.download(model_url, model_path, quiet=False)
            show_info_message("Model weights downloaded successfully.")
        except Exception as e:
            show_error_message(f"Failed to download the model weights: {e}")
            return False
    else:
        show_info_message("Model weights found, skipping download.")
    return True

# Function to create the model architecture
def create_model():
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to load model weights
def load_model_with_weights(model_path):
    try:
        # Create the model architecture
        model = create_model()

        # Load the weights
        model.load_weights(model_path)
        show_info_message("Model weights loaded successfully.")
        return model
    except Exception as e:
        show_error_message(f"Error loading model weights: {e}")
        return None

# Preprocessing function
def pre_process(image_path):
    img = cv2.imread(image_path)
    img_to_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labTran = cv2.cvtColor(img_to_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(labTran)
    claheTran = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = claheTran.apply(l)
    merged = cv2.merge((cl, a, b))
    imgNotResized = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    imgResized = cv2.resize(imgNotResized, IMG_SIZE)
    imgResized = imgResized / 255.0  # Normalize the image
    return imgResized

# Function to load patient images
def load_patient_images(patient_dir):
    if not os.path.exists(patient_dir):
        show_error_message(f"The directory '{patient_dir}' does not exist.")
        return []

    image_paths = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_paths) == 0:
        show_error_message(f"No images found in the directory '{patient_dir}'.")
        return []

    return image_paths

# Function to make predictions for all patient images and apply majority voting
def predict_patient_images(model, image_paths):
    predictions = []
    original_images = []

    show_info_message(f"Found {len(image_paths)} images for the patient. Starting prediction...")

    for image_path in image_paths:
        orig_img = cv2.imread(image_path)
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        processed_img = pre_process(image_path)

        # Make a prediction
        prediction = model.predict(np.expand_dims(processed_img, axis=0))
        label_idx = (prediction > 0.5).astype(int)[0][0]
        predictions.append(label_idx)
        original_images.append(orig_img_rgb)

    # Majority vote to determine final diagnosis
    vote_result = Counter(predictions).most_common(1)[0][0]
    final_label = mapping[vote_result]

    return original_images, predictions, final_label

# Visualize predictions
def visualize_patient_predictions(original_images, predictions, final_label):
    num_images = len(original_images)
    
    # Dynamically determine the number of rows and columns
    cols = 4  # Maximum number of columns
    rows = math.ceil(num_images / cols)  # Calculate rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f'Patient Diagnosis: {final_label}', fontsize=20)

    # Flatten axes for easier indexing (handles 1D or 2D axes arrays)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        if i < num_images:
            # Display image and prediction
            ax.imshow(original_images[i])
            predicted_label = mapping[predictions[i]]
            ax.set_title(f'Predicted: {predicted_label}', fontsize=14)
            
            # Add a colored border based on prediction
            color = 'green' if predictions[i] == 0 else 'red'
            rect = Rectangle((0, 0), original_images[i].shape[1], original_images[i].shape[0],
                             linewidth=5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        else:
            # Hide any extra axes
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# Main function to run the inference process
def main():
    # Download the model weights if necessary
    if not download_model_weights(MODEL_PATH, MODEL_URL):
        show_error_message("Model weights download failed. Exiting.")
        return

    # Load the model with weights
    model = load_model_with_weights(MODEL_PATH)
    if model is None:
        show_error_message("Failed to load the model. Exiting.")
        return

    # Load patient images
    image_paths = load_patient_images(PATIENT_DIR)
    if not image_paths:
        show_error_message("No valid images found. Exiting.")
        return

    # Run predictions and visualize results
    original_images, predictions, final_label = predict_patient_images(model, image_paths)
    show_info_message(f"Final diagnosis for the patient based on majority vote: {final_label}")

    # Display the results
    visualize_patient_predictions(original_images, predictions, final_label)

# Run the main function
if __name__ == "__main__":
    main()
