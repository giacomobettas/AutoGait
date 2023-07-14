import joblib
from src.detector import detect_anomalies
from src.alerting import display_and_alert

# Define test video path
test_video_path = 'data/test_video.mp4'

# Load the saved autoencoder model
saved_autoencoder = joblib.load('models/autoencoder_model.pkl')

# Detect anomalies using the saved autoencoder model
anomalies = detect_anomalies(test_video_path, saved_autoencoder)

# Display alerts
display_and_alert(anomalies)
# display_and_alert(anomalies, 'recipient@example.com')
