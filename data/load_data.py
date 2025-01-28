import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_cifar10():
    # Loading CIFAR-10 dataset from TensorFlow Keras API
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Convert the data to a format suitable for saving to CSV
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    train_data = np.column_stack((y_train, X_train_flat))
    test_data = np.column_stack((y_test, X_test_flat))

    # Saving the data to CSV files(For ease of use, sharing and storage)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_csv('train_data.csv', index=False, header=False)
    test_df.to_csv('test_data.csv', index=False, header=False)

    # Path to save visualizations
    visualizations_file = 'reports/visualizations.png'
    
    # Figure for the visualizations
    fig = plt.figure(figsize=(20, 10))
    
    # EDA: Visualizing data by plotting images
    # This is an initial exploration step to gain insights into the data's characteristics and quality
    ax1 = fig.add_subplot(1, 2, 1)
    k = 0
    for i in range(5):
        for j in range(5):
            ax1 = fig.add_subplot(5, 5, k+1)
            ax1.imshow(X_train[k])
            ax1.axis('off')
            k += 1
    
    # EDA: Class distribution in training set
    # Exploring class distribution is valuable for assessing class balance and identifying any potential biases 
    ax2 = fig.add_subplot(1, 2, 2)
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    classes, counts = np.unique(y_train, return_counts=True)

    ax2.barh(class_names, counts)
    ax2.set_title('Class distribution in training set')
    ax2.set_xlabel('Number of Images')
    
    # Save the combined visualizations
    plt.savefig(visualizations_file)
    plt.show()

    return X_train, y_train, X_test, y_test

# Calling the function
X_train, y_train, X_test, y_test = load_cifar10()
