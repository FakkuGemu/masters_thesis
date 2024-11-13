import os

import numpy as np
from scipy.io import wavfile

def lms_filter(desired_signal, noise_signal, mu=0.01, filter_order=32):
    """
    Implementacja filtru LMS.
    desired_signal - sygnał z mikrofonu głównego (sygnał + szum)
    noise_signal - sygnał szumu z mikrofonu referencyjnego
    mu - współczynnik uczenia (szybkość adaptacji)
    filter_order - długość filtru (liczba współczynników)

    Zwraca: przefiltrowany sygnał (sygnał po redukcji szumów)
    """
    n_samples = len(desired_signal)
    
    w = np.zeros(filter_order)
    buffer = np.zeros(filter_order)
    filtered_signal = np.zeros(n_samples)

    for i in range(n_samples):
        
        buffer[1:] = buffer[:-1]
        buffer[0] = noise_signal[i]

        
        predicted_noise = np.dot(w, buffer)
        error = desired_signal[i] - predicted_noise

        
        w += 2 * mu * error * buffer

        
        filtered_signal[i] = error

    return filtered_signal



folder_path = r'/noise_cancelation_all_methods'
files = [f for f in os.listdir(folder_path) if f.endswith('.wav') and f.startswith('main_audio')]

for file_name in files:
    env_name = file_name.replace("main","env")
    sample_rate, main_signal = wavfile.read(f'noise_cancelation_all_methods\\{file_name}')
    _, noise_signal = wavfile.read(f'noise_cancelation_all_methods\\{env_name}')
    
    if main_signal.dtype == np.int16:
        main_signal = main_signal / 32768.0
        noise_signal = noise_signal / 32768.0
    
    filtered_audio = lms_filter(main_signal, noise_signal, mu=0.001, filter_order=32)
    filtered_audio = np.int16(np.clip(filtered_audio * 32768, -32768, 32767))
    
    save_file = env_name.replace("env_audio", "filtered_LMS")
    wavfile.write(f'noise_cancelation_all_methods\\{save_file}', sample_rate, np.int16(filtered_audio))
