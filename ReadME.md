# AI-Based Rodent Infestation Detection System

## Overview
The AI-Based Rodent Infestation Detection System is a smart monitoring solution designed to detect rodent activity in grain storage environments using acoustic analysis and deep learning techniques. The system continuously captures environmental audio through a microphone and analyzes the sound patterns using a CNN-based deep learning model to identify rodent activities such as scratching, gnawing, rustling, and movement.

When rodent activity is detected, the system activates an alert mechanism using a buzzer and LED through an ESP32-based IoT simulation.

---

# Features

- Real-time rodent activity detection
- Acoustic monitoring using microphone input
- CNN-based audio classification
- MFCC feature extraction
- Continuous environmental monitoring
- Automatic buzzer and LED alerts
- ESP32 IoT simulation using Wokwi
- Environmental sound filtering

---

# Technologies Used

## Artificial Intelligence & Machine Learning
- Python
- TensorFlow
- Keras
- CNN (Convolutional Neural Network)
- MFCC Feature Extraction
- NumPy
- Scikit-learn

## Audio Processing
- Librosa
- PyDub
- SoundDevice
- SoundFile
- FFmpeg

## IoT & Embedded Systems
- ESP32
- Arduino Framework
- Wokwi Simulator

## Alert System
- Pygame
- Buzzer
- LED Indicator

---

# Project Structure

```text
rodent-infest/
│
├── dataset/
│   ├── normal/
│   └── rodent/
│
├── model/
│   └── cnn_lstm_model.h5
│
├── train_model.py
├── realtime_detection.py
├── requirements.txt
├── alert.mp3
└── README.md
```

---

# Dataset Preparation

The dataset is divided into two folders:

## Normal Sounds
Store environmental sounds such as:
- Fan sounds
- Room ambience
- AC sounds
- Keyboard typing
- Environmental noise

Inside:
```text
dataset/normal/
```

## Rodent Sounds
Store rodent activity sounds such as:
- Scratching
- Gnawing
- Rustling
- Rodent movement
- Chewing sounds

Inside:
```text
dataset/rodent/
```

All audio files should preferably:
- Be in WAV format
- Be around 3 seconds long
- Be mono audio

---

# Installation

## Step 1 — Clone Repository

```bash
git clone <repository-url>
cd rodent-infest
```

## Step 2 — Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# How to Start the Project

## Step 1 — Prepare Dataset

Place all audio files inside:

```text
dataset/normal/
dataset/rodent/
```

Make sure:
- All files are WAV format
- Audio clips are around 3 seconds

---

## Step 2 — Train the AI Model

Run:

```bash
python train_model.py
```

The system will:
1. Load dataset
2. Extract MFCC features
3. Train CNN model
4. Save trained model

After training completes:

```text
MODEL SAVED SUCCESSFULLY
```

The trained model will be saved in:

```text
model/cnn_lstm_model.h5
```

---

## Step 3 — Run Real-Time Detection

Run:

```bash
python realtime_detection.py
```

The system will:
- Continuously listen through microphone
- Capture environmental audio
- Extract MFCC features
- Predict rodent activity
- Trigger buzzer alert if rodent activity is detected

---

# How to Test the System

1. Run the real-time detection script.
2. Play rodent scratching or gnawing sounds through a phone speaker.
3. Keep the phone near the microphone.
4. Observe predictions in terminal.

Example:

```text
Prediction: Rodent
Confidence: 92.15%

RODENT DETECTED!
```

---

# Wokwi IoT Simulation

The project also includes an ESP32-based IoT simulation using Wokwi.

## Components Used
- ESP32
- Buzzer
- LED
- Push Button

## Simulation Logic
- Push button simulates rodent detection
- LED glows during detection
- Buzzer activates during detection

---

# Working Principle

1. Microphone captures environmental audio.
2. Audio is converted into MFCC features.
3. CNN model classifies audio patterns.
4. If rodent activity is identified:
   - Buzzer activates
   - LED turns ON
   - Detection message displayed

---

# Future Enhancements

- Raspberry Pi implementation
- GSM/Wi-Fi notifications
- Mobile application integration
- Cloud-based monitoring
- Improved deep learning models
- Ultrasonic rodent detection

---

# Keywords

Rodent Detection, CNN-LSTM, Acoustic Monitoring, MFCC, Deep Learning, IoT, ESP32, Real-Time Detection, Audio Classification, Smart Grain Storage

---

# Authors

Developed as an academic mini-project for smart grain storage monitoring and rodent infestation prevention.