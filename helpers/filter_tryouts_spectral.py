import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft


sample_rate, main_signal = wavfile.read('../noise_cancelation_all_methods/main_audio1.wav')
_, noise_signal = wavfile.read('../noise_cancelation_all_methods/env_audio1.wav')
folder_path = r'/noise_cancelation_all_methods'
files = [f for f in os.listdir(folder_path) if f.endswith('.wav') and f.startswith('main_audio')]

for file_name in files:
    env_name = file_name.replace("main","env")
    sample_rate, main_signal = wavfile.read(f'noise_cancelation_all_methods\\{file_name}')
    _, noise_signal = wavfile.read(f'noise_cancelation_all_methods\\{env_name}')
    f_1, t_1, Sxx_1 = stft(main_signal, fs=sample_rate)
    f_2, t_2, Sxx_2 = stft(noise_signal, fs=sample_rate)

    magnitude_1 = np.abs(Sxx_1)
    magnitude_2 = np.abs(Sxx_2)
    phase_1 = np.angle(Sxx_1)

    magnitude_cleaned = np.maximum(magnitude_1 - magnitude_2, 0)

    Zxx_cleaned = magnitude_cleaned * np.exp(1j * phase_1)

    _, cleaned_audio = istft(Zxx_cleaned, fs=sample_rate)
    save_file = env_name.replace("env_audio", "filtered_spectral")
    wavfile.write(f'noise_cancelation_all_methods\\{save_file}', sample_rate, np.int16(cleaned_audio))

