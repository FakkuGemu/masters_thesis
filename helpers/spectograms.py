import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


filename = '../noise_cancelation_files/env_audio_no_delay_threshold,04_dB,67.wav'  
y, sr = librosa.load(filename)


S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)


S_dB = librosa.power_to_db(S, ref=np.max)


plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-spektrogram')
plt.show()
