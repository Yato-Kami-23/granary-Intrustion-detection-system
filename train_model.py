import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import Adam

DATASET_PATH = "dataset"

labels = ["normal", "rodent"]

X = []
y = []

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        duration=3,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = mfcc[:, :130]

    if mfcc.shape[1] < 130:

        pad_width = 130 - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0,0),(0,pad_width)),
            mode='constant'
        )

    return mfcc

for label_index, label in enumerate(labels):

    folder = os.path.join(DATASET_PATH, label)

    print(f"\nLoading {label} sounds...")

    for file in os.listdir(folder):

        if file.endswith(".wav"):

            file_path = os.path.join(folder, file)

            try:

                features = extract_features(file_path)

                X.append(features)
                y.append(label_index)

                print(f"Loaded: {file}")

            except Exception as e:

                print(f"Error loading {file}")
                print(e)

X = np.array(X)
y = np.array(y)

print("\nDataset Loaded Successfully")

print("Feature Shape:", X.shape)

X = X.reshape(X.shape[0], 40, 130, 1)

y = to_categorical(y, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nBuilding Model...")

model = Sequential()

model.add(
    Conv2D(
        16,
        (3,3),
        activation='relu',
        input_shape=(40,130,1)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nStarting Training...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_data=(X_test, y_test)
)

os.makedirs("model", exist_ok=True)

model.save("model/cnn_lstm_model.h5")

print("\nMODEL SAVED SUCCESSFULLY")