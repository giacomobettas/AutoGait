import joblib
# import sys
from src.autoencoder import train_autoencoder
from src.detector import detect_anomalies
from src.alerting import display_and_alert


if __name__ == 'main':
    # Define the video paths
    input_folder_path = 'data/input_videos'
    test_video_path = 'data/test_video.mp4'

    # Train the autoencoder and save the model
    autoencoder = train_autoencoder(input_folder_path, 'models/autoencoder_model.pkl')

    """
    # Choose the autoencoder model to use from command line
    if len(sys.argv) < 2:
        print("Provide the path to the saved model file.")
        sys.exit(1)

    model_path = sys.argv[1]
    """

    # Load the saved autoencoder model
    saved_autoencoder = joblib.load('models/autoencoder_model.pkl')

    # Detect anomalies using the saved autoencoder model
    anomalies = detect_anomalies(test_video_path, saved_autoencoder)

    # Display alerts
    display_and_alert(anomalies)
    # display_and_alert(anomalies, 'recipient@example.com')
