import os
from src.autoencoder import train_autoencoder
from tensorflow.keras.models import load_model

# Define the input videos folder
input_folder_path = 'data/input_videos'

# Train the autoencoder and save the model
autoencoder_model_path = 'models/autoencoder_model.h5'
train_autoencoder(input_folder_path, autoencoder_model_path)

# Use the saved autoencoder for further processing (e.g., anomaly detection)
