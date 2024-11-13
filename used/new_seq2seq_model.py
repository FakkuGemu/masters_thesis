import os
import glob
import librosa
import numpy as np
from scipy.signal import resample
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, TimeDistributed, Flatten, LSTM, Dense, \
    Masking


learn_data_folder = r'C:\Users\ivank\Desktop\magsiterka\pythonProject\oddech\learn'
validation_data_folder = r'C:\Users\ivank\Desktop\magsiterka\pythonProject\oddech\walidate'
test_data_folder = r'C:\Users\ivank\Desktop\magsiterka\pythonProject\oddech\test'
generated_data_folder = r'C:\Users\ivank\Desktop\magsiterka\pythonProject\oddech\generated'



def load_audio(file_path, sr=22050, target_length=2100000):
    y, sr = librosa.load(file_path, sr=sr)

    
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), 'constant')

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram.T  



def load_labels(file_path, sr=22050, target_length=2100000):
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
    
    mask = np.zeros(target_length)
    for (start, end, label) in labels:
        mask[start:end] = label
    return mask



def pad_sequences_custom(sequences, maxlen=None, padding_value=0):
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



def prepare_data(data_folder, sr=22050, target_length=2100000):
    audio_files = glob.glob(os.path.join(data_folder, '*.wav'))
    label_files = [f.replace('.wav', '.txt') for f in audio_files]

    X = []
    y = []

    for audio_file, label_file in zip(audio_files, label_files):
        
        spectrogram = load_audio(audio_file, sr=sr, target_length=target_length)
        X.append(spectrogram)


        
        label_mask = load_labels(label_file, sr=sr, target_length=target_length)
        y.append(label_mask)

    
    X = pad_sequences_custom(X)  
    y = pad_sequences_custom(y, padding_value=0)  

    
    X = X[..., np.newaxis]  
    print(X)
    print(y)

    return X, y



def build_seq2seq_model(input_shape, output_length):
    inputs = Input(shape=input_shape)

    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    x = UpSampling2D(size=(2, 1))(x)

    
    x = TimeDistributed(Flatten())(x)

    
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)

    
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



def rescale_labels(labels, target_len):
    rescaled_labels = []
    for label in labels:
        rescaled_label = resample(label, target_len)
        
        rescaled_label = (rescaled_label > 0.5).astype(int)
        rescaled_labels.append(rescaled_label)
    return np.array(rescaled_labels)



print("Przygotowanie danych do pretrainingu...")
X_pretrain, y_pretrain = prepare_data(generated_data_folder)

print("Przygotowanie danych do walidacji...")
X_val, y_val = prepare_data(validation_data_folder)

print("Przygotowanie danych do treningu...")
X_train, y_train = prepare_data(learn_data_folder)

print("Przygotowanie danych do testowania...")
X_test, y_test = prepare_data(test_data_folder)








input_shape = X_train.shape[1:]  
output_length = y_train.shape[1]  

model = build_seq2seq_model(input_shape, output_length)
model.summary()


print("Reskalowanie etykiet...")
y_pretrain_rescaled = rescale_labels(y_pretrain, output_length)
y_val_rescaled = rescale_labels(y_val, output_length)
y_train_rescaled = rescale_labels(y_train, output_length)
y_test_rescaled = rescale_labels(y_test, output_length)

print(f"input_shape: {input_shape}")
print(f"output_length : {output_length }")

print("Rozpoczęcie pretrainingu na danych wygenerowanych...")
model.fit(X_pretrain, y_pretrain_rescaled, epochs=10, batch_size=16, validation_data=(X_val, y_val_rescaled))

print("Rozpoczęcie finetuningu na rzeczywistych danych...")
model.fit(X_train, y_train_rescaled, epochs=20, batch_size=16, validation_data=(X_val, y_val_rescaled))


model.save('oddech_model_finetuned.h5')
print("Model zapisany jako 'oddech_model_finetuned.h5'")


print("Testowanie modelu na danych testowych...")
loaded_model = tf.keras.models.load_model('oddech_model_finetuned.h5')


y_test_rescaled = rescale_labels(y_test, output_length)


test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test_rescaled)
print(f'Loss na danych testowych: {test_loss}')
print(f'Accuracy na danych testowych: {test_accuracy}')