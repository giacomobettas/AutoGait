import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Reshape, Conv2DTranspose
from src.video_utils import load_video_frames

def train_autoencoder(input_folder_path, save_path):
    # Load video frames from the folder
    frames = load_video_frames(input_folder_path)

    # Convert frames to numpy array (2D)
    X_train = np.array(frames)

    # Create and train the autoencoder model (CNN-based)
    autoencoder = Sequential()

    # Encoder
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(240, 320, 1)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # autoencoder.add(Flatten())

    # Decoder
    # autoencoder.add(Reshape(X_train.shape[1:]))
    autoencoder.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'))

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=1)

    # Save the trained autoencoder model to a file
    autoencoder.save(save_path)

    return autoencoder
