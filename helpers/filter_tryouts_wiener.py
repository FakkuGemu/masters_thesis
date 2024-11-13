import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft


sample_rate, main_signal = wavfile.read('../noise_cancelation_all_methods/main_audio1.wav')
_, noise_signal = wavfile.read('../noise_cancelation_all_methods/env_audio1.wav')
folder_path = r'/noise_cancelation_all_methods'
files = [f for f in os.listdir(folder_path) if f.endswith('.wav') and f.startswith('main_audio')]


n_fft = 1024
hop_length = 512
window = 'hann'
for file_name in files:
    env_name = file_name.replace("main","env")
    sample_rate, main_signal = wavfile.read(f'noise_cancelation_all_methods\\{file_name}')
    _, noise_signal = wavfile.read(f'noise_cancelation_all_methods\\{env_name}')
    
    _, _, Zxx_main = stft(main_signal, fs=sample_rate, nperseg=n_fft, noverlap=hop_length, window=window)
    _, _, Zxx_noise = stft(noise_signal, fs=sample_rate, nperseg=n_fft, noverlap=hop_length, window=window)

    
    SNR = np.abs(Zxx_main) / (np.abs(Zxx_noise) + 1e-10)  
    wiener_gain = SNR / (1 + SNR)  

    
    filtered_Zxx = Zxx_main * wiener_gain

    
    _, denoised_audio = istft(filtered_Zxx, fs=sample_rate, nperseg=n_fft, noverlap=hop_length, window=window)
    save_file = env_name.replace("env_audio", "filtered_wiener")
    wavfile.write(f'noise_cancelation_all_methods\\{save_file}', sample_rate, np.int16(denoised_audio))