import os
import sys
import numpy as np
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from tensorflow.keras.utils import to_categorical

from data.load_data import load_cifar10

def preprocess_data(X_train, X_test, y_train, y_test):
    # Normalizing pixel values to be between 0 and 1 (It ensures a consistent range contributing to model convergence and stability during training.)
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # One-hot encode the labels (Converting into binary vectors easier for processing during training.)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Flatten the data for saving to CSV
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Combine labels and features into one array
    train_data = np.column_stack((y_train, X_train_flat))
    test_data = np.column_stack((y_test, X_test_flat))

    # Save the data to CSV files
    np.savetxt('train_data.csv', train_data, delimiter=',')
    np.savetxt('test_data.csv', test_data, delimiter=',')
    
    # Authenticating and creating the PyDrive client
    print("Authenticating with Google Drive...")
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Google Drive folder ID
    folder_id = '1YSYybxyPR8yRAqgJEt8QZMLrgBiXEq_y'  

    # Upload the CSV files directly to Google Drive
    print("Uploading preprocessed data to Google Drive...")
    upload_file_to_drive(drive, 'train_data.csv', folder_id)
    upload_file_to_drive(drive, 'test_data.csv', folder_id)

def upload_file_to_drive(drive, file_name, folder_id):
    file = drive.CreateFile({'title': file_name, 'parents': [{'id': folder_id}]})
    file.SetContentFile(file_name)
    file.Upload()
    print(f"Uploaded {file_name} to Google Drive.")
    # Remove local file after upload
    os.remove(file_name)
    print(f"Removed local file {file_name}.")

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_cifar10()
    preprocess_data(X_train, X_test, y_train, y_test)
