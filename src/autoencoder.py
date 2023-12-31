import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from src.video_utils import load_video_frames

def train_autoencoder(input_folder_path, save_path):
    # Load video frames from the folder
    frames = load_video_frames(input_folder_path)
    
    # Convert frames to numpy array and flatten each frame
    X_train = np.array(frames)
    num_frames, height, width = X_train.shape
    X_train = X_train.reshape(num_frames, -1)
    
    # Create and train the autoencoder model
    autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 128),
                               activation='relu', random_state=49, max_iter=1000,
                               early_stopping=True, verbose=True)
    autoencoder.fit(X_train, X_train)

    # Save the trained autoencoder model to a file
    joblib.dump(autoencoder, save_path)

    return autoencoder
