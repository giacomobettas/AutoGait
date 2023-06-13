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

# Alert function
def send_alert():
    cv2.imshow('Anomaly Detected!', np.zeros((100, 300), dtype=np.uint8)) # Displaying a pop-up window using OpenCV
    cv2.waitKey(3000)  # Display the alert for 3 seconds
    cv2.destroyAllWindows()

# Sound alert function
def play_sound():
    from playsound import playsound
    sound_file_path = 'path_to_sound_file.mp3'  # Path to sound file
    playsound(sound_file_path)

# Email notification function with image attachment
def send_email_notification(anomalous_frame):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage

    # Email configuration
    sender_email = 'your_email@gmail.com'
    sender_password = 'your_email_password'
    receiver_email = 'receiver_email@gmail.com'

    # Email content
    subject = 'Anomaly Detected!'
    body = 'Anomalous frame detected. Please check.'

    # Create the email message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    # Attach the anomalous frame as an image
    frame_attachment = MIMEImage(anomalous_frame)
    frame_attachment.add_header('Content-Disposition', 'attachment', filename='anomalous_frame.jpg')
    message.attach(frame_attachment)

    # Send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)

# Anomaly detection loop
for frame in validation_frames:
    reconstructed_frame = autogait.predict(np.expand_dims(frame, axis=0))
    error = np.mean(np.square(frame - reconstructed_frame))
    
    if error > threshold:
        anomalous_frames.append(frame)
        play_sound()
        send_email_notification()
