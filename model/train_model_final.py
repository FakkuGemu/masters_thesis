import os
import glob
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


vggish_model = hub.load("https://tfhub.dev/google/vggish/1")


learn_data_folder = r'C:\Users\ivank\Desktop\magisterka3.10\pythonProject\oddech\learn'
validation_data_folder = r'C:\Users\ivank\Desktop\magisterka3.10\pythonProject\oddech\walidate'


SAMPLE_RATE = 44100
SEGMENT_DURATION = 0.96  
TARGET_LENGTH = int(SAMPLE_RATE * SEGMENT_DURATION)  


def extract_vggish_features(audio_signal, sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION):
    segment_length = int(segment_duration * sample_rate)
    features = []

    
    for start in range(0, len(audio_signal), segment_length):
        segment = audio_signal[start:start + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')

        vggish_features = vggish_model(segment)
        features.append(vggish_features.numpy().squeeze())

    return np.array(features)  


def load_labels(file_path, audio_length, sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION):
    labels = np.zeros(audio_length, dtype=int)  

    
    with open(file_path, 'r') as f:
        for line in f:
            label_type, start, end = line.strip().split()
            start_sample, end_sample = int(float(start)), int(float(end))
            label = 1 if label_type == 'wdech' else 2  
            labels[start_sample:end_sample] = label

    
    segment_length = int(segment_duration * sample_rate)
    segment_labels = [
        np.bincount(labels[start:start + segment_length]).argmax()
        for start in range(0, len(labels), segment_length)
    ]
    return np.array(segment_labels)



def prepare_data(data_folder, sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION):
    audio_files = glob.glob(os.path.join(data_folder, '*.wav'))
    label_files = [f.replace('.wav', '.txt') for f in audio_files]

    X, y = [], []
    for audio_file, label_file in zip(audio_files, label_files):
        y_segment, _ = librosa.load(audio_file, sr=sample_rate)

        
        features = extract_vggish_features(y_segment, sample_rate=sample_rate, segment_duration=segment_duration)

        
        labels = load_labels(label_file, audio_length=len(y_segment), sample_rate=sample_rate,
                             segment_duration=segment_duration)

        X.extend(features)
        y.extend(labels)
    
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)  

    y = np.array(y)
    y = tf.keras.utils.to_categorical(y, num_classes=3)

    return np.expand_dims(X, axis=0), np.expand_dims(y, axis=0)



def build_model(input_shape, num_classes=3):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = layers.LSTM(64, return_sequences=True, dropout=0.3)(x)
    outputs = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model



X_train, y_train = prepare_data(learn_data_folder)
X_val, y_val = prepare_data(validation_data_folder)
print(X_train)
print(y_train)

input_shape = (None, X_train.shape[2], X_train.shape[3])
model = build_model(input_shape)
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=64,
          callbacks=[early_stopping, reduce_lr])

model.save('breathing_model_inhales2.h5')
