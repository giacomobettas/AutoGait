import cv2
import numpy as np

def detect_anomalies(test_video_path, saved_autoencoder):

    cap = cv2.VideoCapture(test_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    anomalies = []
    threshold = 0.05  # Adjust the threshold value based on requirements

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reshape the frame to match the autoencoder's input shape
        frame = frame.reshape(1, frame.shape[0], frame.shape[1], 1)
        """ The frame is reshaped to (1, height, width, 1),
            where 1 is for the number of samples (batch size),
            height and width are the dimensions of the frame,
            and 1 is for the number of channels (grayscale) """

        # Predict the reconstructed frame using the autoencoder
        reconstructed_frame = saved_autoencoder.predict(frame)

        # Calculate the reconstruction error
        error = np.mean(np.square(frame - reconstructed_frame))

        if error > threshold:
            anomalies.append(frame)

    cap.release()
    return anomalies
