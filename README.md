# music_genre_classification
This project aims to classify music genres using machine learning techniques. We utilize the VGGish model from TensorFlow Hub to extract features from audio files and train a convolutional neural network (CNN) to predict the genre of a given music clip.
Steps and Code Explanation
# 1. Mount Google Drive
Mount Google Drive to access the dataset stored in the drive.


from google.colab import drive
drive.mount('/content/drive')
# 2. Import Necessary Libraries
Import all the required libraries for data preprocessing, visualization, and model building.


import numpy as np
import pandas as pd
import os, librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow_hub as hub
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import layers, Sequential
from keras.callbacks import EarlyStopping

from warnings import filterwarnings
filterwarnings('ignore')
# 3. Load VGGish Model from TensorFlow Hub
Load the pre-trained VGGish model for feature extraction.

vggish = hub.load('https://tfhub.dev/google/vggish/1')
# 4. Define Path of Audio Files
Specify the path where the audio files are stored.


data_path = '/content/drive/MyDrive/gtzan_music_genre_classification/Data/genres_original'
# 5. Display Waveform and Spectrogram of Sample Music from Each Genre
Visualize the waveform and spectrogram of the first audio file from each genre.

genres = os.listdir(data_path)

genre_colors = {
    'blues': 'blue',
    'classical': 'green',
    'country': 'red',
    'disco': 'purple',
    'hiphop': 'orange',
    'jazz': 'pink',
    'metal': 'brown',
    'pop': 'cyan',
    'reggae': 'lime',
    'rock': 'gray',
}

genre_colormaps = {
    'blues': 'viridis',
    'classical': 'plasma',
    'country': 'inferno',
    'disco': 'magma',
    'hiphop': 'cividis',
    'jazz': 'twilight',
    'metal': 'twilight_shifted',
    'pop': 'YlOrBr',
    'reggae': 'YlOrRd',
    'rock': 'YlGnBu',
}

for genre in genres:
    genre_path = os.path.join(data_path, genre)
    if os.path.isdir(genre_path):
        audio_files = os.listdir(genre_path)
        audio_file = os.path.join(genre_path, audio_files[0])

        waveform, sample_rate = librosa.load(audio_file)

        print(f'Class: {genre.capitalize()}\n')
        display(Audio(waveform, rate=sample_rate))

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(waveform, color=genre_colors.get(genre, 'blue'))
        plt.title('Waveform', fontsize=16)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)

        plt.subplot(1, 2, 2)
        plt.specgram(waveform, Fs=sample_rate, cmap=genre_colormaps.get(genre, 'viridis'))
        plt.title('Spectrogram', fontsize=16)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)

        plt.show()
# 6. Define a Function to Extract Audio Features using VGGish Model
Create a function to load audio files, preprocess them, and extract features using the VGGish model.


def extractFeatures(audioFile):
    try:
        waveform, sr = librosa.load(audioFile)
        waveform, _ = librosa.effects.trim(waveform)
        return vggish(waveform).numpy()
    except:
        return None
# 7. Process Audio Files to Extract Features
Iterate through all audio files, extract features using the defined function, and store them in a DataFrame.


root = '/content/drive/MyDrive/gtzan_music_genre_classification/Data/genres_original'
data = []

for folder in os.listdir(root):
    folderPath = os.path.join(root, folder)
    for file in tqdm(os.listdir(folderPath), desc=f'Processing folder {folder}'):
        filePath = os.path.join(folderPath, file)
        features = extractFeatures(filePath)
        if features is not None:
            data.append([features, folder])

data = pd.DataFrame(data, columns=['Features', 'Class'])
data.head()
# 8. Convert Data to DataFrame and Store
Save the extracted features and labels to a CSV file.


data.to_csv('Music_genre_df.csv', index=False)


# 9. Padding
Pad or truncate the feature arrays to a fixed length for uniform input to the model.


x = data['Features'].tolist()
x = pad_sequences(x, dtype='float32', padding='post', truncating='post')
x.shape
# 10. Encoding
Encode the class labels into categorical format for model training.


encoder = LabelEncoder()
y = encoder.fit_transform(data['Class'])
y = to_categorical(y)
# 11. Train, Validation, Test Split
Split the data into training and testing sets.


trainX, testX, trainY, testY = train_test_split(x, y, random_state=0)
# 12. Early Stopping
Implement early stopping to prevent overfitting during training.


earlyStopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)
# 13. Model Fitting
Define and train the CNN model.


model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(43, 128, 1)),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=50,
    batch_size=32,
    callbacks=[earlyStopping]
)

historyDf = pd.DataFrame(history.history)
historyDf.to_csv('Music_genre_history_df.csv', index=False)
# 14. Model Evaluation
Evaluate the model's performance and plot the training history.


historyDf.loc[:, ['loss', 'val_loss']].plot()
historyDf.loc[:, ['accuracy', 'val_accuracy']].plot()
# 15. Model Saving
Save the trained model for future use.


model.save('Music_genre_model.h5')
# 16. Prediction
Define functions to preprocess audio files and make predictions using the trained model.


def extract_features(audio_file):
    try:
        waveform, sr = librosa.load(audio_file)
        waveform, _ = librosa.effects.trim(waveform)
        features = vggish(waveform).numpy()
        return features
    except:
        return None

def preprocess_audio(audio_file_path):
    features = extract_features(audio_file_path)
    if features is not None:
        features = np.expand_dims(features, axis=-1)
        features = np.expand_dims(features, axis=0)
        features = pad_sequences(features, maxlen=43, padding='post', truncating='post', dtype='float32')
        return features
    else:
        return None

def predict_genre(model, audio_file_path):
    preprocessed_data = preprocess_audio(audio_file_path)
    if preprocessed_data is not None:
        prediction = model.predict(preprocessed_data)
        predicted_label = encoder.inverse_transform([np.argmax(prediction)])
        print(f'Predicted Genre: {predicted_label[0]}')
    else:
        print('Error preprocessing the audio file.')

external_audio_file_path = '/content/drive/MyDrive/gtzan_music_genre_classification/Data/genres_original/blues/blues.00015.wav'
display(Audio(external_audio_file_path))
predict_genre(model, external_audio_file_path)
