import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

folder_path = r"/noise_cancelation_all_methods"

all_files = os.listdir(folder_path)

suffixes = set(file_name.split('_', 1)[1] for file_name in all_files if file_name.endswith("audio4.wav"))

for suffix in suffixes:
    file_set = [
        f"{prefix}_{suffix}"
        for prefix in ['main', 'env', 'filtered']
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for i, file_name in enumerate(file_set):
        audio_path = os.path.join(folder_path, file_name)

        if not os.path.exists(audio_path):
            print(f"File {audio_path} does not exist. Skipping.")
            continue

        y, sr = librosa.load(audio_path, sr=None)  

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_dB, sr=sr, ax=axes[i], x_axis='time', y_axis='mel', cmap='viridis')
        axes[i].set_title(f"Spectrogram of {file_name}")

        axes[i].set_ylim([0, sr // 2])

        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Frequency (Hz)")

        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')

    plt.tight_layout()

    output_file = os.path.join(folder_path, f"spectrograms_{suffix.replace('.wav', '')}.png")
    plt.savefig(output_file)