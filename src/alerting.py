import cv2
import numpy as np
# from utils.email_sender import send_email


def display_and_alert(anomalies, email_recipient):
    for frame in anomalies:
        frame = frame.reshape(480, 640).astype(np.uint8)

        # Display the frame
        cv2.imshow('Anomaly Detected', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save the frame as an image
        cv2.imwrite('test/anomaly.jpg', frame)

    cv2.destroyAllWindows()
