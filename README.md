# Image Segmentation Using U-Net

This project implements **semantic segmentation** using the **U-Net architecture**, focused on analyzing aerial or satellite imagery to detect forested areas. It supports both **image** and **video** inputs, using a trained deep learning model.

[Open in Google Colab](https://colab.research.google.com/drive/1tBKoZH-HV0F8APzJiP9wA1O0fU4UeOOw?usp=sharing)

---

## 📚 Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [U-Net Architecture](#u-net-architecture)  
- [Colab Workflow](#colab-workflow)  
- [Image Segmentation](#image-segmentation)  
- [Video Segmentation](#video-segmentation)  
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

U-Net is a convolutional neural network designed for biomedical and environmental image segmentation.

### Architecture Highlights:
- **Encoder**: Convolution + MaxPooling
- **Bottleneck**: Deep features
- **Decoder**: Transposed Convolution + Skip Connections

The model captures spatial context and reconstructs pixel-level masks.

---

## 🚀 Colab Workflow

Everything runs in a Colab notebook:

- Load and preprocess data
- Build U-Net model
- Train and validate
- Save model (`model.h5`)
- Run predictions on new data (images or videos)

👉 [Launch Colab Notebook](https://colab.research.google.com/drive/1tBKoZH-HV0F8APzJiP9wA1O0fU4UeOOw?usp=sharing)

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

### Metrics (example):
- Dice Coefficient: `0.87`
- IoU Score: `0.78`

---

## 📦 Dependencies

Make sure to install:

```bash
tensorflow
opencv-python
numpy
matplotlib
