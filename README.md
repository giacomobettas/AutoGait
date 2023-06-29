# AutoGait

This project performs video anomaly detection using an autoencoder trained with a video containing normal frames. It utilizes the scikit-learn library to develop the autoencoder for anomaly detection. The project includes an alerting section that displays the anomalous video frames on the screen and sends an alert via email with attached images of the detected anomalies.

## Project Structure

The project is structured as follows:

    AutoGait/
    ├── data/
    │ ├── normal_video.mp4
    │ └── test_video.mp4
    ├── src/
    │ ├── autoencoder.py
    │ ├── detector.py
    │ └── alerting.py
    ├── test/
    │ ├── anomaly.jpg
    ├── utils/
    │ └── email_sender.py
    ├── Project_lineup.md
    ├── README.md
    ├── main.py
    └── requirements.txt


The main components of the project are:

- `data/`: This directory contains the video files used for training the autoencoder and detecting anomalies. The `normal_video.mp4` file is used for training, and the `test_video.mp4` file is used for anomaly detection.
- `src/`: This directory contains the Python scripts that implement the autoencoder, detector, and alerting functionality.
- `test/`: This directory contains the detector outputs.
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
4. Run the project:

    Run `main.py` script to train the autoencoder, detect anomalies, and display/send alerts.

    ```plaintext
    $ python main.py

5. Evaluate the results:

    - Monitor the console output for any detected anomalies during the execution of the main script.
    - The script will display anomalous video frames on the screen.