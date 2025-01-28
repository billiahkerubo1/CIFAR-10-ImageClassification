import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_architecture import create_baseline_cnn, create_vgg16_model

# Load the preprocessed data from CSV files
train_data = pd.read_csv('preprocessed_train.csv')
test_data = pd.read_csv('preprocessed_test.csv')

# Separate features and labels
X_train = train_data.iloc[:, 10:].values  
y_train = train_data.iloc[:, :10].values
X_test = test_data.iloc[:, 10:].values
y_test = test_data.iloc[:, :10].values

# Reshape the feature arrays to match the input shape for the models
X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

# Train the Baseline CNN Model
print("Training Baseline CNN Model...")
baseline_model = create_baseline_cnn(input_shape=(32, 32, 3), num_classes=10)
history_baseline = baseline_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Train the VGG16 Model
print("Training VGG16 Model...")
vgg16_model = create_vgg16_model(input_shape=(32, 32, 3), num_classes=10)
history_vgg16 = vgg16_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Plot Training Histories
plt.figure(figsize=(12, 6))
plt.plot(history_baseline.history['accuracy'], label='Baseline Train Accuracy')
plt.plot(history_baseline.history['val_accuracy'], label='Baseline Val Accuracy')
plt.plot(history_vgg16.history['accuracy'], label='VGG16 Train Accuracy')
plt.plot(history_vgg16.history['val_accuracy'], label='VGG16 Val Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
