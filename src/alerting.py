import os
import cv2
import numpy as np
# from utils.email_sender import send_email

def alert(anomalies, output_root_folder):

    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder) # Create the output root folder if it doesn't exist

    for i, anomaly in enumerate(anomalies):
        anomaly_folder = os.path.join(output_root_folder, f'anomaly_{i + 1}') # Create a folder for each anomaly
        os.makedirs(anomaly_folder, exist_ok=True)

        # Save the first frame of the anomaly
        first_frame = anomaly[0]

        first_frame = first_frame.reshape(240, 320).astype(np.uint8) # Reshape the frame to its original dimensions
        # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR) # Convert to BGR format

        first_frame_name = 'first_frame.jpg'
        first_frame_path = os.path.join(anomaly_folder, first_frame_name)
        cv2.imwrite(first_frame_path, first_frame)

        # Save the last frame of the anomaly
        last_frame = anomaly[-1]

        last_frame = last_frame.reshape(240, 320).astype(np.uint8) # Reshape the frame to its original dimensions
        # last_frame = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR) # Convert to BGR format

        last_frame_name = 'last_frame.jpg'
        last_frame_path = os.path.join(anomaly_folder, last_frame_name)
        cv2.imwrite(last_frame_path, last_frame)

"""     for frame in anomalies:
        
        # Display the frame
        cv2.imshow('Anomaly Detected', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save the frame as an image
        cv2.imwrite('test/anomaly.jpg', frame)

    cv2.destroyAllWindows() """
