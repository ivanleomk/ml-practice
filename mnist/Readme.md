# MNIST Handwritten Digit Recognition

This folder contains the necessary Python scripts to train and evaluate a neural network model on the MNIST dataset, a collection of handwritten digits.

## Installation

Before running the scripts, ensure you have Python installed on your system. Then, follow these steps to set up your environment:

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source venv/bin/activate
     ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Scripts

1. **Download the MNIST dataset**: The `download.py` script downloads the MNIST dataset and stores it in a local directory. Run the following command:

   ```
   python download.py
   ```

   This will create a `data` directory with `train` and `test` subdirectories containing the dataset.

2. **Train the model**: The `train.py` script trains a neural network on the MNIST training dataset. To start training, run:
   ```
   python train.py
   ```
   This script initializes a neural network model, trains it on the dataset, and prints the training loss and accuracy.

## Overview

- `download.py`: Downloads the MNIST dataset and prepares it for training.
- `train.py`: Defines a neural network model, trains it on the MNIST dataset, and evaluates its performance.

The neural network defined in `train.py` consists of a sequence of layers designed to recognize patterns in the MNIST images. It uses a combination of linear transformations and non-linear activations (ReLU) to process the input images. The training process involves optimizing the model's parameters to minimize the difference between the predicted and actual labels of the images.
