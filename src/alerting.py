import cv2
import numpy as np
# from utils.email_sender import send_email

def display_and_alert(anomalies):
    for frame in anomalies:
        frame = frame.reshape(240, 320).astype(np.uint8) # Reshape the frame to its original dimensions
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Convert to BGR format

        # Display the frame
        cv2.imshow('Anomaly Detected', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save the frame as an image
        cv2.imwrite('test/anomaly.jpg', frame)

    cv2.destroyAllWindows()
