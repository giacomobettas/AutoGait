from src.detector import detect_anomalies
from src.alerting import alert
from tensorflow.keras.models import load_model

# Define test video path
test_video_path = 'data/test_video.mp4'

# Load the saved autoencoder model
autoencoder_model_path = 'models/autoencoder_model.h5'
saved_autoencoder = load_model(autoencoder_model_path)

# Detect anomalies using the saved autoencoder model
anomalies = detect_anomalies(test_video_path, saved_autoencoder)
# Print the number of anomalies detected
print(f'Anomalies detected: {len(anomalies)}')

# Display alerts
output_root_folder = 'test'  # Specify your desired output folder
alert(anomalies, output_root_folder)
