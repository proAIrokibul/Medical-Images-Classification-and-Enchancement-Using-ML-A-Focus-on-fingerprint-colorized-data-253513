# Medical-Images-Classification-and-Enchancement-Using-ML-A-Focus-on-fingerprint-colorized-data-253513
## 🔍 Project Overview

This project aims to explore machine learning and deep learning techniques for classifying and enhancing **medical fingerprint images** using a colorized dataset derived from **FVC2000 DB4_B**. The goal is to evaluate different model architectures to achieve accurate and efficient classification of fingerprint data across various enhancement techniques.

The dataset comprises 13 enhancement folders (e.g., Gaussian Blur, Heatmap, Edge Detection, etc.), each containing images representing fingerprint samples processed with a specific visualization or augmentation technique.

---

## 📁 Dataset

- **Source**: FVC2000 Fingerprint Dataset (DB4_B)
- **Path**: `train_data/Fingerprint Dataset for FVC2000_DB4_B/train_data`
- **Structure**:  
  13 folders including:
  - `3D_Rendering`
  - `Adaptive_Histogram_Equalization`
  - `Gaussian_Blur`
  - `Heatmap_Visualization`
  - `Edge_Detection`
  - and others...
- Each folder contains colorized images of fingerprint samples.

---

## 📈 Business Impact

This project holds significant value in multiple real-world domains:

### 🏥 **Medical Biometrics**
- Enhancing and accurately classifying fingerprint images can improve **patient identification systems** in hospitals and clinics.
- Prevents medical errors caused by patient ID mismatches.
- Enables **secure access control** for sensitive areas or digital health records.

### 🔐 **Cybersecurity & Access Control**
- Integrating fingerprint classification models into access systems enhances **security and fraud detection**.
- Useful in **multimodal authentication systems** where biometric accuracy is critical.

### 🕵️ **Forensics & Law Enforcement**
- Faster classification can speed up **criminal identification workflows**.
- Useful in matching latent prints with high-confidence predictions even after enhancement.

### 🌍 **Scalable Solutions for Developing Regions**
- Low-compute models like MLP and custom CNNs can be deployed in resource-constrained settings (rural clinics, portable biometric devices).
- Promotes **inclusivity in digital identity systems**.

---

## 🧹 Preprocessing

- Resized all images to `128x128` pixels
- Normalized pixel values
- Applied `train/validation` split (80/20)
- Converted data using `ImageFolder` and PyTorch `DataLoader`

---

## 📊 Exploratory Data Analysis (EDA)

- Visualized sample images from each category
- Displayed class distributions
- Checked data balance across enhancement techniques
- Plotted pixel-level histograms

---

## 🧠 Models Implemented

### ✅ 1. **MLP (Multilayer Perceptron)**  
- Fast and shallow model  
- Flattened image inputs  
- Dense layers with dropout

### ✅ 2. **Custom CNN**  
- Lightweight convolutional architecture  
- 3 conv blocks → FC layers  
- Quick training and robust performance

### ✅ 3. **ResNet18 (Offline-Compatible)**  
- Pretrained-weights removed for offline use  
- Deep residual architecture  
- Best performance among all models

---

## 📈 Model Comparison

| Model         | Validation Accuracy |
|---------------|---------------------|
| MLP           | 59.52%              |
| Custom CNN    | 57.07%              |
| ResNet18      | 63.61%              |

- 📊 Bar chart included to visualize performance
- 🔍 ResNet18 outperforms others but all models can benefit from further tuning

