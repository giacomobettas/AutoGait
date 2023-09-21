# AutoGait

This project performs video anomaly detection using an autoencoder trained with a video containing normal frames. It utilizes the Keras library to develop the CNN-based autoencoder for anomaly detection. The project includes an alerting section that saves the anomalous video frames in a directory and sends an alert via email with attached images of the detected anomalies.

This initial stage consists of developing an autoencoder that can be used to recognize when a person falls. Afterwards, the ability to recognize a person by their gait and distinguish different dangerous situations in a domestic environment will be added. The ultimate goal is to monitor the elderly automatically in their home environment.

## Project Structure

The project is structured as follows:

    AutoGait/
    ├── data/
    │ └── input_videos
    │   ├── normal_video1.mp4
    │   └── normal_video2.mp4
    │ └── test_video.mp4
    ├── models/
      └── autoencoder_model.h5
    ├── src/
    │ ├── alerting.py
    │ ├── autoencoder.py
    │ ├── detector.py
    │ └── video_utils.py
    ├── test/
    │ ├── anomaly_1
    │   ├── first_frame.jpg
    │   └── last_frame.jpg
    │ ├── anomaly_2
    │   ├── first_frame.jpg
    │   └── last_frame.jpg
    │ ├── anomaly_3
    │   ├── first_frame.jpg
    │   └── last_frame.jpg
    ├── utils/
    │ └── email_sender.py
    ├── Project_lineup.md
    ├── README.md
    ├── autoencoder_train.py
    ├── main.py
    └── requirements.txt


The main components of the project are:

- `data/`: This directory contains the video files used for training the autoencoder and detecting anomalies. The `input_videos/` directory contains normal videos used for training, while the `test_video.mp4` file is used for anomaly detection.
- `models/`: This folder contains the autoencoder trained models. They can be loaded into the script to run tests with different models.
- `src/`: This directory contains the Python scripts that implement the autoencoder, detector, and alerting functionality.
- `test/`: This directory contains the detector outputs. There will be a folder for any anomaly detected containing the first and the last frame of the respective anomaly.
- `utils/`: This directory contains a utility script for sending email alerts with attachments (work in progress).
- `Project_lineup.md`: This file explains how to organize and implement the anomaly detection project applied to the gait.
- `README.md`: This file provides an overview of the project, instructions for setting up and running the project, and other relevant information.
- `requirements.txt`: This file lists the dependencies required to run the project.

## Steps

1. Set up a virtual environment:

    ```plaintext
    $ python3 -m venv venv
    $ source venv/bin/activate

2. Install the dependencies:

    ```plaintext
    $ pip install -r requirements.txt

3. Fine tune the project:

    - Replace videos in the `data/` directory.
    - Adjust the autoencoder architecture, threshold, and other parameters.
&nbsp;  
&nbsp;  
4. Train the autoencoder:

    ```plaintext
    $ python3 autoencoder_train.py

5. Run the project:

    Run `main.py` script to detect anomalies and display/send alerts.

    ```plaintext
    $ python3 main.py
    
6. Evaluate the results:

    - Monitor the console output for any detected anomalies during the execution of the main script.
    - The script will display anomalous video frames on the screen and save them in the `test/` directory.
    - Fine tune the autoencoder and the detector in order to obtain the expected result.
