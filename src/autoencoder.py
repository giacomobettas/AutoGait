from sklearn.neural_network import MLPRegressor
import joblib
import numpy as np


def train_autoencoder(X_train, save_path):
    # Preprocess and reshape the input video frames
    X_train = X_train.reshape(X_train.shape[0], -1)

    # Create and train the autoencoder model
    autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 128), activation='relu', max_iter=1000)
    autoencoder.fit(X_train, X_train)

    # Save the trained autoencoder model to a file
    joblib.dump(autoencoder, save_path)

    return autoencoder
