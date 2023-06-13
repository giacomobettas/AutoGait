# Import Libraries

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from AutoGait_train import preprocess_frames # Call the preprocessing function


# Load the autoencoder model
autogait = keras.models.load_model('path_to_autoencoder_model.h5')

# Validation and Threshold Selection

validation_video_path = 'validation_video.mp4'
validation_frames = preprocess_frames(validation_video_path)

# Calculate reconstruction error for each validation frame
validation_frames = np.array(validation_frames)
reconstructed_frames = autogait.predict(validation_frames)
reconstruction_errors = np.mean(np.square(validation_frames - reconstructed_frames), axis=(1, 2, 3))

# Analyze error distribution and set threshold for anomaly detection
threshold = np.percentile(reconstruction_errors, 95)  # Example: Set threshold at the 95th percentile


# Anomaly Detection

anomalous_frames = []
for frame in validation_frames:
    reconstructed_frame = autogait.predict(np.expand_dims(frame, axis=0))
    error = np.mean(np.square(frame - reconstructed_frame))
    
    if error > threshold:
        anomalous_frames.append(frame)

# Convert anomalous frames back to BGR color format for visualization
anomalous_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in anomalous_frames]


# Post-processing and Visualization

# Display anomalous frames
for frame in anomalous_frames:
    cv2.imshow('Anomaly Detection', frame)
    cv2.waitKey(1000)  # Show each frame for 1 second
cv2.destroyAllWindows()

# Alerts
