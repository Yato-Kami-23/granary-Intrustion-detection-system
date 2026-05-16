import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import pygame
import time

MODEL_PATH = "model/cnn_lstm_model.h5"

print("=" * 50)
print("AI RODENT INFESTATION DETECTION SYSTEM")
print("=" * 50)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully")

labels = ["Normal", "Rodent"]

pygame.mixer.init()

# CHANGE THIS TO YOUR MIC DEVICE
MIC_DEVICE = 2

def extract_mfcc(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = mfcc[:, :130]

    if mfcc.shape[1] < 130:

        pad = 130 - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0,0),(0,pad)),
            mode='constant'
        )

    return mfcc

def record_audio(duration=3, sr=22050):

    print("\nListening...")

    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype='float32',
        device=MIC_DEVICE
    )

    sd.wait()

    audio = audio.flatten()

    volume = np.max(np.abs(audio))

    print(f"Volume Level: {volume:.4f}")

    return audio, sr

while True:

    try:

        audio, sr = record_audio()

        mfcc = extract_mfcc(audio, sr)

        mfcc = mfcc.reshape(1,40,130,1)

        prediction = model.predict(
            mfcc,
            verbose=0
        )

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        label = labels[predicted_class]

        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence * 100:.2f}%")

        if label == "Rodent" and confidence > 0.65:

            print("\nRODENT DETECTED!")

            try:

                pygame.mixer.music.load("alert.mp3")
                pygame.mixer.music.play()

            except Exception as e:

                print(e)

        time.sleep(1)

    except KeyboardInterrupt:

        print("\nSystem stopped")
        break

    except Exception as e:

        print("\nError occurred:")
        print(e)

        time.sleep(1)