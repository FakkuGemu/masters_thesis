import os
import librosa
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Dense, LSTM, UpSampling2D


def load_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram.T


def pad_sequences(sequences, maxlen=None, padding_value=0):
    if maxlen is None:
        maxlen = max([s.shape[0] for s in sequences])  

    padded_sequences = []
    for seq in sequences:
        if seq.shape[0] < maxlen:
            pad_width = maxlen - seq.shape[0]
            if len(seq.shape) == 1:  
                
                padded_seq = np.pad(seq, (0, pad_width), mode='constant', constant_values=padding_value)
            else:
                
                padded_seq = np.pad(seq, ((0, pad_width), (0, 0)), mode='constant', constant_values=padding_value)
        else:
            
            padded_seq = seq[:maxlen]

        padded_sequences.append(padded_seq)

    return np.array(padded_sequences)


def load_labels(file_path, sr=22050):
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            label_type = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            label = 1 if label_type == 'wdech' else 2

            labels.append((start, end, label))
    return labels



def create_label_mask(audio_len, labels, sr=22050):
    mask = np.zeros(audio_len)
    print(audio_len)
    for (start, end, label) in labels:
        start_frame = int(start / sr)
        end_frame = int(end / sr)
        mask[start_frame:end_frame] = label
    return mask



def prepare_data(data_folder):
    audio_files = glob.glob(os.path.join(data_folder, '*.wav'))
    label_files = [f.replace('.wav', '.txt') for f in audio_files]
    print(audio_files)
    print(label_files)

    X = []
    y = []

    for audio_file, label_file in zip(audio_files, label_files):
        
        spectrogram = load_audio(audio_file)
        print(spectrogram)
        X.append(spectrogram)

        
        labels = load_labels(label_file)
        label_mask = create_label_mask(len(spectrogram), labels)
        print(label_mask)
        y.append(label_mask)

    
    X = pad_sequences(X)  
    y = pad_sequences(y, padding_value=0)  

    
    X = X[..., np.newaxis]

    return X, y


def build_seq2seq_model(input_shape):
    model = tf.keras.Sequential()

    
    model.add(Input(shape=input_shape))

    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    
    model.add(UpSampling2D(size=(2, 1)))  

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(TimeDistributed(Flatten()))

    
    model.add(LSTM(128, return_sequences=True))

    
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))  

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



learn_data_folder = r'C:\Users\ivank\Desktop\magsiterka\pythonProject\oddech\learn'
validation_data_folder = r'C:\Users\ivank\Desktop\magsiterka\pythonProject\oddech\walidate'
test_data_folder = r'C:\Users\ivank\Desktop\magsiterka\pythonProject\oddech\test'

generated_data_folder = r'C:\Users\ivank\Desktop\magsiterka\pythonProject\oddech\generated'


X_pretrain, y_pretrain = prepare_data(generated_data_folder)


X_train, y_train = prepare_data(learn_data_folder)
X_val, y_val = prepare_data(validation_data_folder)


print(f"X train shape: {X_pretrain.shape[1:]}")
model = build_seq2seq_model(X_pretrain.shape[1:])


print("Rozpoczęcie pretrainingu na danych wygenerowanych...")
model.fit(X_pretrain, y_pretrain, epochs=10, batch_size=16, validation_data=(X_val, y_val))


print("Rozpoczęcie finetuningu na rzeczywistych danych...")
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))




loaded_model = tf.keras.models.load_model('oddech_model_finetuned.h5')


X_test, y_test = prepare_data(test_data_folder)


test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test)
print(f'Loss na danych testowych: {test_loss}')
print(f'Accuracy na danych testowych: {test_accuracy}')