from src.autoencoder import train_autoencoder
from src.detector import detect_anomalies
from src.alerting import display_and_alert


# Define the video paths
normal_video_path = 'data/normal_video.mp4'
test_video_path = 'data/test_video.mp4'

# Train the autoencoder
# Load the normal video frames for training
X_train = load_video_frames(normal_video_path)
autoencoder = train_autoencoder(X_train)

# Detect anomalies in the test video
anomalies = detect_anomalies(test_video_path, autoencoder)

# Display alerts
display_and_alert(anomalies)
# display_and_alert(anomalies, 'recipient@example.com')
