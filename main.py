from src.video_utils import load_video_frames
from src.autoencoder import train_autoencoder
from src.detector import detect_anomalies
from src.alerting import display_and_alert
import joblib


if __name__ == 'main':
    # Define the video paths
    normal_video_path = 'data/normal_video.mp4'
    test_video_path = 'data/test_video.mp4'

    # Load the normal video frames for training
    X_train = load_video_frames(normal_video_path)
    # Train the autoencoder and save the model
    autoencoder = train_autoencoder(X_train, 'models/autoencoder_model.pkl')

    # Save the trained autoencoder model
    joblib.dump(autoencoder, 'models/autoencoder_model.pkl')

    """
    # Choose the autoencoder model to use from command line
    if len(sys.argv) < 2:
        print("Please provide the path to the saved model file.")
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
