# CIFAR10 Image Classification using TensorFlow

## Overview
The project aims to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, that include airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The project implements two different approaches that include a custom-designed CNN architecture built from scratch to serve as a performance benchmark and a transfer-learning model that uses the VGG16 model pretrained on the ImageNet dataset. It fine-tunes it for the CIFAR-10 dataset to leverage the rich feature representations learned from a larger and more diverse dataset.

## Key Features:
- **Data Loading and Preprocessing**: Scripts to load the CIFAR-10 dataset, normalize the images, and split the data into training, validation, and test sets.
- **Model Development**: Scripts and Jupyter notebooks to build, train, and evaluate both the baseline CNN model and the transfer learning model.
- **Model Comparison**: A dedicated script to evaluate and compare the performance of the two models using common metrics such as accuracy, precision, recall, and F1-score.
- **Web Application Deployment**: The trained model is deployed as a web application using Flask, allowing users to upload images and receive classification results in real-time.

## Objectives:
1. **Build and Evaluate Baseline CNN Model**: Develop a custom CNN model for CIFAR-10 image classification and establish performance benchmarks.
2. **Implement Transfer Learning with VGG16**: Fine-tune the VGG16 model for CIFAR-10 classification and compare its performance with the baseline model.
3. **Deploy Model as a Web Application**: Create a user-friendly web interface for real-time image classification and deploy it using Flask.

## Setup Instructions

### Prerequisites
- Python 3.8 or later
- Git

### Installation
Clone the repository:
```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository

