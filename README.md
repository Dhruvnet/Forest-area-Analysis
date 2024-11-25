# Image Segmentation Using U-Net

This project implements **image segmentation** using the **U-Net architecture**, leveraging **Google Colab** for training and evaluation. The dataset is directly loaded from Kaggle, making it easy to replicate and customize the workflow. It focuses on segmenting images by generating pixel-wise masks.

[Open the Colab Notebook](https://colab.research.google.com/drive/1tBKoZH-HV0F8APzJiP9wA1O0fU4UeOOw?usp=sharing)

---

## Table of Contents

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [U-Net Architecture](#u-net-architecture)  
4. [Colab Workflow](#colab-workflow)  
5. [Usage](#usage)  
6. [Results](#results)  
7. [Dependencies](#dependencies)  

---

## Overview

Image segmentation is a process of dividing an image into multiple meaningful regions. This project uses **U-Net**, a convolutional neural network architecture widely used for segmentation tasks, especially in biomedical and environmental imaging.

The Colab notebook:
- Preprocesses input images and masks.
- Builds the U-Net model architecture using TensorFlow/Keras.
- Trains the model on the Kaggle dataset.
- Evaluates and visualizes segmentation results.

---

## Dataset

The dataset used for this project is the **Augmented Forest Segmentation Dataset**, which is available on Kaggle:  
[Download the Dataset from Kaggle](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation).

In the Colab notebook:
- The dataset is directly downloaded and installed using Kaggle’s API.
- Images and masks are resized to `128x128` pixels for uniformity.
- Masks are normalized and binarized for segmentation tasks.

**Note**: Ensure you have a Kaggle API token to access the dataset programmatically. Instructions are included in the Colab notebook.

---

## U-Net Architecture

The **U-Net** model has a symmetric encoder-decoder structure:
1. **Encoder**: Extracts spatial features using convolutional layers and max-pooling.
2. **Bottleneck**: Captures the most abstracted feature representations.
3. **Decoder**: Reconstructs segmentation maps using transposed convolutions and skip connections.

The notebook:
- Implements U-Net using TensorFlow/Keras.
- Provides flexibility to adjust input shape, filters, and output channels for different segmentation tasks.

---

## Colab Workflow

The Colab notebook is structured as follows:

### 1. Dataset Setup
- Automatically downloads the dataset from Kaggle using the Kaggle API.
- Processes the dataset by resizing images and masks to `128x128`.

### 2. Data Preprocessing
- Normalizes image pixel values to `[0, 1]`.  
- Converts masks to binary format for segmentation.

### 3. Model Definition
- Builds U-Net with TensorFlow/Keras.
- Employs skip connections for spatial detail retention.

### 4. Model Training
- Compiles the model using **Binary Crossentropy Loss** and **Adam Optimizer**.  
- Displays real-time loss and accuracy graphs during training.

### 5. Visualization
- Predicts segmentation masks for test images.
- Displays input images, true masks, and predicted masks side by side.

---

## Usage

### Steps to Use the Notebook:
1. [Open the Colab Notebook](https://colab.research.google.com/drive/1tBKoZH-HV0F8APzJiP9wA1O0fU4UeOOw?usp=sharing).  
2. Set up Kaggle API credentials in the notebook for dataset access.
3. Run the notebook cells sequentially to:
   - Download and preprocess the dataset.
   - Train the U-Net model.
   - Visualize segmentation results.

### Running Locally
To run the project outside Colab:
1. Clone this repository.
2. Download the dataset manually from Kaggle and extract it locally.
3. Install dependencies (see [Dependencies](#dependencies)).
4. Run the `.ipynb` file in Jupyter Notebook or VS Code with Python kernel.

---

## Results

The Colab notebook visualizes:
- **Input Images**  
- **Ground Truth Masks**  
- **Predicted Segmentation Masks**

These outputs demonstrate the trained U-Net model's segmentation accuracy.

---

## Dependencies

The following Python libraries are required:
- TensorFlow >= 2.4  
- NumPy  
- Matplotlib  

Install them using:
```bash
pip install tensorflow numpy matplotlib
