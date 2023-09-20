import os
import cv2

def load_video_frames(input_folder_path):
    frames = []

    # Accept a folder path as input
    # Iterate through the video files in the folder and load their frames
    for filename in os.listdir(input_folder_path):
        video_path = os.path.join(input_folder_path, filename)

        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

        cap.release()
    return frames
