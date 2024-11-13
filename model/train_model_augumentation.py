import os
import glob
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.ndimage import shift
import random



vggish_model = hub.load("https://tfhub.dev/google/vggish/1")


learn_data_folder = r'C:\Users\ivank\Desktop\magisterka3.10\pythonProject\oddech\learn'
validation_data_folder = r'C:\Users\ivank\Desktop\magisterka3.10\pythonProject\oddech\walidate'

SAMPLE_RATE = 44100
SEGMENT_DURATION = 2.0  
TARGET_LENGTH = int(SAMPLE_RATE * SEGMENT_DURATION)  

def extract_vggish_features(audio_signal, sample_rate=44100, segment_duration=0.96):
    segment_length = int(segment_duration * sample_rate)
    features = []

    
    for start in range(0, len(audio_signal), segment_length):
        segment = audio_signal[start:start + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')

        vggish_features = vggish_model(segment)
        features.append(vggish_features.numpy().squeeze())

    return np.array(features)  


def load_labels(file_path, segment_duration=0.96, total_length=None):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            label_type, start, end = line.strip().split()
            start, end = float(start), float(end)
            label = 1 if label_type == 'wdech' else 2  
            start_segment = int(start / segment_duration)
            end_segment = int(end / segment_duration)
            labels.extend([label] * (end_segment - start_segment + 1))

    if total_length and len(labels) < total_length:
        labels.extend([0] * (total_length - len(labels)))  
    return np.array(labels)


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def speed_tune(data, rate=1.0):
    return librosa.effects.time_stretch(data.astype(float), rate=rate)


def pitch_shift(data, sample_rate, n_steps=0):
    return librosa.effects.pitch_shift(data.astype(float), sr=sample_rate, n_steps=n_steps)


def time_shift(data, shift_max=0.2):
    shift = int(shift_max * len(data) * random.choice([-1, 1]))
    return np.roll(data, shift), shift


def stretch_labels(labels, rate):
    
    new_length = int(len(labels) / rate)
    stretched_labels = np.round(np.interp(np.linspace(0, len(labels) - 1, new_length), np.arange(len(labels)), labels))
    return stretched_labels.astype(int)


def shift_labels(labels, shift):
    
    if shift > 0:
        return np.pad(labels, (shift, 0), 'constant')[:len(labels)]
    elif shift < 0:
        return np.pad(labels, (0, -shift), 'constant')[-len(labels):]
    return labels


def augment_audio(audio, labels, sample_rate):
    augmented_audio, augmented_labels = [], []

    
    speed_rate = np.random.uniform(0.9, 1.1)
    audio_speed_tuned = librosa.effects.time_stretch(audio, rate=speed_rate)
    
    new_labels_speed = librosa.effects.time_stretch(labels.astype(float), rate=speed_rate).round().astype(int)
    augmented_audio.append(audio_speed_tuned)
    augmented_labels.append(new_labels_speed)

    
    pitch_shift = np.random.randint(-2, 3)
    audio_pitch_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
    augmented_audio.append(audio_pitch_shifted)
    augmented_labels.append(labels)

    
    max_shift = int(0.2 * sample_rate)  
    shift_offset = np.random.randint(-max_shift, max_shift)
    audio_time_shifted = shift(audio, shift_offset, mode='nearest')
    new_labels_time_shifted = shift(labels, shift_offset, mode='nearest')
    augmented_audio.append(audio_time_shifted)
    augmented_labels.append(new_labels_time_shifted)

    
    noise_factor = np.random.uniform(0.002, 0.01)
    noise = np.random.randn(len(audio)) * noise_factor
    audio_noisy = audio + noise
    augmented_audio.append(audio_noisy)
    augmented_labels.append(labels)

    return augmented_audio, augmented_labels

def prepare_data(data_folder, sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION):
    audio_files = glob.glob(os.path.join(data_folder, '*.wav'))
    label_files = [f.replace('.wav', '.txt') for f in audio_files]

    X, y = [], []
    for audio_file, label_file in zip(audio_files, label_files):
        y_segment, _ = librosa.load(audio_file, sr=sample_rate)

        if len(y_segment) < TARGET_LENGTH:
            y_segment = np.pad(y_segment, (0, TARGET_LENGTH - len(y_segment)), 'constant')
        else:
            y_segment = y_segment[:TARGET_LENGTH]

        features = extract_vggish_features(y_segment, sample_rate=SAMPLE_RATE)
        labels = load_labels(label_file, total_length=features.shape[0])

        X.append(features)
        y.append(labels)

        
        augmented_audio, augmented_labels = augment_audio(y_segment, labels, sample_rate)
        for aug_audio, aug_label in zip(augmented_audio, augmented_labels):
            aug_features = extract_vggish_features(aug_audio, sample_rate=SAMPLE_RATE)
            if aug_features.shape[0] == features.shape[0]:  
                X.append(aug_features)
                y.append(aug_label)

    max_length = max([f.shape[0] for f in X])
    X = np.array([
        np.pad(f, ((0, max_length - f.shape[0]), (0, 0)), 'constant') if f.shape[0] < max_length else f[:max_length]
        for f in X
    ])
    y = np.array([
        np.pad(lbl, (0, max_length - len(lbl)), 'constant') if len(lbl) < max_length else lbl[:max_length]
        for lbl in y
    ])

    y = tf.keras.utils.to_categorical(y, num_classes=3)  
    return X, y




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

input_shape = X_train.shape[1:]
model = build_model(input_shape)
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=64,
          callbacks=[early_stopping, reduce_lr])

model.save('breathing_checkup.h5')