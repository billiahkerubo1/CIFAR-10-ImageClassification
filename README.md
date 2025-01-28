## CIFAR10 Image Classification using Tensorflow

### Overview
The project aims to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, that include airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The project implements two different approaches that include a custom-designed CNN architecture built from scratch to serve as a performance benchmark and transfer-learning model that uses the VGG16 model pretrained on the ImageNet dataset. It fine-tunes it for the CIFAR-10 dataset to leverage the rich feature representations learned from a larger and more diverse dataset.


Key Features:

Data Loading and Preprocessing: Consists of scripts to load the CIFAR-10 dataset, normalize the images, and split the data into training, validation, and test sets.

Model Development: The project includes scripts and Jupyter notebooks to build, train, and evaluate both the baseline CNN model and the transfer learning model.

Model Comparison: A dedicated script to evaluate and compare the performance of the two models using common metrics such as accuracy, precision, recall, and F1-score.

Web Application Deployment: The trained model is deployed as a web application using Flask, allowing users to upload images and receive classification results in real time. 

Objectives
Build and Evaluate Baseline CNN Model: Develop a custom CNN model for CIFAR-10 image classification and establish performance benchmarks.

Implement Transfer Learning with VGG16: Fine-tune the VGG16 model for CIFAR-10 classification and compare its performance with the baseline model.

Deploy Model as a Web Application: Create a user-friendly web interface for real-time image classification and deploy it using Flask

### Baseline Model Architecture
The CNN architecture includes multiple essential components:
1.  convolutional layers to extract spatial features and patterns from the input data

2. max-pooling layers to down-sample the feature maps and retain the most important features

3. A flattening layer to convert the 2D feature maps into a 1D vector for the fully connected layers, dense (fully connected) layers to perform classification or regression tasks

4. Dropout regularization to prevent overfitting by randomly setting a fraction of the input units to zero during training. Combined

These components enable the CNN to effectively learn and generalize patterns from the data for accurate predictions.

### Setup Instructions

#### Prerequisites
- Python 3.8 or later
- Git

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
### Usage
To train the model, run the train.py script. The trained model will be saved in the model.h5 file. To evaluate the models on the test set, run the model_evaluation.ipynb file . The accuracy of the model on the test set will be printed to the console.