# 🎯 Backend for Emotion Detection Using Audio-Visual Modalities

This Node.js backend powers a React-based frontend for detecting human emotions from both audio and video inputs. Designed with safety-critical environments in mind—like driver monitoring systems—this backend processes inputs, runs AI inference, and returns predicted emotional states.

---

## 📌 Overview

This backend is part of a full-stack application that enables real-time emotion detection using multimodal data (audio + visual). It uses pre-trained and fine-tuned deep learning models to analyze audio features (like MFCCs) and video frames to infer emotions such as happiness, anger, fear, etc.

---

## 🧠 AI Model Details

- **Model Name**: VideoMAE (Masked Autoencoder for Video)
- **Fine-tuned Version**: `TFVideoMAE_L_K400_16x224_FT`
- **Pre-trained on**: Kinetics-400
- **Frameworks Used**: TensorFlow, Keras
- **Inference Mode**: Python (served via Node.js)

---

## 📂 Datasets Used

- 🎧 **CREMA-D**  
  - Includes 7442 labeled audio-visual clips  
  - Labels: Angry, Happy, Neutral, Disgust, Fear, Sad

- 🎥 **eNTERFACE**
  - Multimodal emotion dataset with video and audio clips  
  - Ideal for validating cross-dataset generalization

---

## 🔧 Data Preprocessing and Fusion

### 🎙️ Audio Processing

- **MFCCs** (Mel-Frequency Cepstral Coefficients)
- **Mel Spectrograms**
- Converted to image format for CNN-based feature extraction

### 🎞️ Video Processing

- Video split into fixed-length clips (e.g., 2 seconds)
- Frames extracted at 16 FPS and resized to `224x224`

### 🔗 Fusion Techniques

1. **Early Fusion**:  
   Audio and video features combined before classification.

2. **Late Fusion**:  
   Separate branches for audio and video → merged at decision layer.

---

## 🌐 API Endpoints

| Endpoint              | Method | Description                          |
|-----------------------|--------|--------------------------------------|
| `/upload/audio`       | POST   | Uploads audio file for processing    |
| `/upload/video`       | POST   | Uploads video file for processing    |
| `/predict`            | POST   | Runs inference and returns emotion   |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/backend-emotion-detection.git
cd backend-emotion-detection
