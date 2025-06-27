# Image Segmentation Using U-Net

This project implements **semantic segmentation** using the **U-Net architecture**, focused on analyzing aerial or satellite imagery to detect forested areas. It supports both **image** and **video** inputs, using a trained deep learning model.

---

## 📚 Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [U-Net Architecture](#u-net-architecture)   
- [Image Segmentation](#image-segmentation)  
- [Results](#results)  
- [Dependencies](#dependencies)

---

## 🔍 Overview

The goal is to perform **pixel-wise classification** of aerial images to identify forest areas. This is achieved using a custom-trained **U-Net model** built with **TensorFlow/Keras**.

You can:
- Train on images with masks
- Use the trained model to segment new images or even **videos** frame-by-frame

---

## 🗂️ Dataset

We used the **Augmented Forest Segmentation Dataset** from Kaggle:

🔗 [Download the Dataset from Kaggle](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation)

The dataset contains:
- Aerial/satellite images
- Binary segmentation masks (forest vs. non-forest)

Preprocessing steps:
- Resize all images and masks to `128x128`
- Normalize image pixel values
- Binarize masks

---

## 🧠 U-Net Architecture

Our model follows the classic U-Net architecture with custom modifications to improve feature learning, reduce overfitting, and handle high-resolution imagery better. The U-Net consists of an **encoder-decoder structure** with **skip connections** that combine high-level semantic features with low-level spatial details.

#### 🔹 1. Encoder (Contracting Path)
The encoder progressively captures spatial and contextual features using a series of convolutional layers.

- Each block in the encoder contains:
  - Two **3x3 convolutional layers** (with padding)
  - **ReLU** activation
  - **Batch Normalization**
  - **2x2 Max Pooling** for downsampling
- Feature channels are doubled at each depth level:
  - Example: 64 → 128 → 256 → 512

#### 🔹 2. Bottleneck
The bridge between encoder and decoder.

- Two convolutional layers with ReLU and BatchNorm
- High-depth features (e.g., 1024 filters) learned here

#### 🔹 3. Decoder (Expanding Path)
Upsamples the feature maps and combines them with corresponding encoder features to reconstruct spatial resolution.

- Each block includes:
  - **Transpose Convolution (UpSampling)**
  - **Concatenation** with encoder features (skip connection)
  - Two **3x3 Convolutional layers** with ReLU and BatchNorm

#### 🔹 4. Output Layer
- **1x1 Convolution** to reduce feature map to a single channel (for binary segmentation)
- **Sigmoid Activation** to produce per-pixel probability map (0 to 1)

---

## 🖼️ Segmentation Output

After training, you can input new images and get segmentation masks:

![image](https://github.com/user-attachments/assets/b685cbd2-886f-4672-98d8-fe5ed532b3f0)

---
<!---
## 🎥 Video Segmentation

You can also apply the model to a video. Each frame is processed using the trained U-Net model, and masks are generated frame-by-frame.

📹 **Example Output Video:**

![Video Demo](path-to-output-video.gif)  
*Overlay of original frames with predicted masks*

To process a video:
- Extract frames with OpenCV
- Resize and normalize each frame
- Predict mask using `model.h5`
- Recombine into a video with overlay (or side-by-side)

---
--->

## ✅ Results

The model performs well in distinguishing between forest and non-forest regions. Visual results show sharp boundaries and good generalization even on unseen data.

| **Metric**           | **Score** |
|----------------------|-----------|
| **Accuracy**         | 0.8363    |
| **Dice Coefficient** | 0.8039    |
| **F1 Score**         | 0.8662    |
| **IoU Score**        | 0.6724    |
| **Loss**             | 0.3678    |
| **Precision**        | 0.8417    |
| **Recall**           | 0.8933    |`

---

## 📦 Dependencies

Make sure to install:

```bash
tensorflow
opencv-python
numpy
matplotlib
