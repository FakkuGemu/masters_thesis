import os
import glob
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


model = tf.keras.models.load_model('breathing_model_inhales2.h5', custom_objects={'KerasLayer': tf.keras.layers.Layer})
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")


test_data_folder = r'C:\Users\ivank\Desktop\magisterka3.10\pythonProject\oddech\test'
SAMPLE_RATE = 44100
SEGMENT_DURATION = 0.96  


class_names = {0: 'silence', 1: 'inhale', 2: 'exhale'}

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

def load_labels(file_path, audio_length, segment_duration=0.96):
    labels = np.zeros(int(audio_length / (segment_duration * SAMPLE_RATE)))
    with open(file_path, 'r') as f:
        for line in f:
            label_type, start, end = line.strip().split()
            start, end = int(start), int(end)
            start_segment = int(start / (segment_duration * SAMPLE_RATE))
            end_segment = int(end / (segment_duration * SAMPLE_RATE))
            label = 1 if label_type == 'wdech' else 2
            labels[start_segment:end_segment+1] = label
    return labels

def classify_full_audio(file_path, label_file_path=None, sample_rate=SAMPLE_RATE, segment_duration=0.96):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    features = extract_vggish_features(audio, sample_rate=sample_rate)
    features = np.expand_dims(features, axis=0)  
    features = np.expand_dims(features, axis=-1)  

    predictions = model.predict(features)
    predicted_labels = np.argmax(predictions, axis=-1).flatten()
    predicted_classes = [class_names[label] for label in predicted_labels]

    if label_file_path:
        real_labels = load_labels(label_file_path, len(audio), segment_duration)
        real_classes = [class_names[int(label)] for label in real_labels]
    else:
        real_classes = None

    min_length = min(len(predicted_classes), len(real_classes))
    predicted_classes = predicted_classes[:min_length]
    real_classes = real_classes[:min_length]

    return predicted_classes, real_classes, audio


overall_real = []
overall_predicted = []


test_files = glob.glob(os.path.join(test_data_folder, '*.wav'))
for file in test_files:
    txt_file = file.replace('.wav', '.txt')
    if os.path.exists(txt_file):
        print(f"Testing on file: {file}")
        predicted_classes, real_classes, audio = classify_full_audio(file, txt_file)

        
        accuracy = accuracy_score(real_classes, predicted_classes)
        print(f"Accuracy for {os.path.basename(file)}: {accuracy * 100:.2f}%")

        
        overall_real.extend(real_classes)
        overall_predicted.extend(predicted_classes)

        
        numeric_real_classes = [list(class_names.keys())[list(class_names.values()).index(cls)] for cls in real_classes]
        numeric_predicted_classes = [list(class_names.keys())[list(class_names.values()).index(cls)] for cls in predicted_classes]
        cm = confusion_matrix(numeric_real_classes, numeric_predicted_classes, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {os.path.basename(file)}")
        plt.show()

        
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 1, 1)
        segment_length = int(SAMPLE_RATE * SEGMENT_DURATION)
        for i in range(len(predicted_classes)):
            start = i * segment_length
            end = start + segment_length
            color = 'green' if predicted_classes[i] == real_classes[i] else 'red'
            plt.plot(np.arange(start, end), audio[start:end], color=color)
        plt.title(f"Audio waveform for {os.path.basename(file)} (Green = Correct, Red = Incorrect)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        
        segment_indices = np.arange(len(predicted_classes)) * SEGMENT_DURATION
        plt.subplot(2, 1, 2)
        plt.plot(segment_indices, predicted_classes, 'o-', label='Predicted', color='blue')
        plt.plot(segment_indices, real_classes, 'x-', label='Real', color='red')
        plt.title("Predicted vs Real Classes per Segment")
        plt.xlabel("Segment Time (s)")
        plt.ylabel("Class")
        plt.legend()
        plt.tight_layout()
        plt.show()


overall_accuracy = accuracy_score(overall_real, overall_predicted)
print(f"Overall Accuracy for the entire folder: {overall_accuracy * 100:.2f}%")


numeric_real_overall = [list(class_names.keys())[list(class_names.values()).index(cls)] for cls in overall_real]
numeric_predicted_overall = [list(class_names.keys())[list(class_names.values()).index(cls)] for cls in overall_predicted]
cm_overall = confusion_matrix(numeric_real_overall, numeric_predicted_overall, labels=[0, 1, 2])
disp_overall = ConfusionMatrixDisplay(confusion_matrix=cm_overall, display_labels=list(class_names.values()))
disp_overall.plot(cmap=plt.cm.Blues)
plt.title("Overall Confusion Matrix for All Test Files")
plt.show()
