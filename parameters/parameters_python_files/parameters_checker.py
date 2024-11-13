import os
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.ndimage import gaussian_filter1d
from itertools import product

from LTA_both_nostrils_bonus_info import start_time


def load_audio(file_path):
    rate, data = wav.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]
    return rate, data


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def smooth_signal(data, sigma=10):
    return gaussian_filter1d(data, sigma=sigma)


def identify_phases(smoothed_data):
    phases = []
    current_phase = None

    
    if smoothed_data[0] > 0:
        current_phase = {'start': 0, 'end': None, 'type': 'wydech', 'values': [smoothed_data[0]]}

    for i in range(1, len(smoothed_data)):
        if smoothed_data[i] > 0 and smoothed_data[i - 1] == 0:
            if current_phase is not None:
                phases.append(current_phase)
            current_phase = {'start': i, 'end': None, 'type': 'wydech', 'values': []}

        if current_phase:
            current_phase['values'].append(smoothed_data[i])

        if smoothed_data[i] == 0 and smoothed_data[i - 1] > 0 and current_phase:
            current_phase['end'] = i
            phases.append(current_phase)
            current_phase = None

    if current_phase and current_phase['end'] is None:
        current_phase['end'] = len(smoothed_data) - 1
        phases.append(current_phase)

    return phases


def adjust_phases(phases, breath_threshold_factor):
    for phase in phases:
        phase['avg'] = np.mean(phase['values'])

    wdechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wdech'])
    wydechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wydech'])

    if np.isnan(wdechy_avg):
        wdechy_avg = 0
    if np.isnan(wydechy_avg):
        wydechy_avg = 0

    threshold = (wdechy_avg + wydechy_avg) / breath_threshold_factor

    for phase in phases:
        if phase['avg'] > threshold:
            phase['type'] = 'wydech'
        else:
            phase['type'] = 'wdech'


def filter_breaths(phases):
    filtered_phases = [phase for phase in phases if phase['type'] in ['wdech', 'wydech']]
    return filtered_phases


def keep_longest_inhale_between_exhales(phases):
    new_phases = []
    current_inhales = []

    for phase in phases:
        if phase['type'] == 'wydech':
            
            if current_inhales:
                
                longest_inhale = max(current_inhales, key=lambda p: p['end'] - p['start'])
                new_phases.append(longest_inhale)  
                
                for inhale in current_inhales:
                    if inhale != longest_inhale:
                        inhale['type'] = 'brak wdechu'
                        new_phases.append(inhale)
                current_inhales = []
            new_phases.append(phase)  
        elif phase['type'] == 'wdech':
            current_inhales.append(phase)

    
    if current_inhales:
        longest_inhale = max(current_inhales, key=lambda p: p['end'] - p['start'])
        new_phases.append(longest_inhale)
        for inhale in current_inhales:
            if inhale != longest_inhale:
                inhale['type'] = 'brak wdechu'
                new_phases.append(inhale)

    return new_phases


def save_phases_to_file(phases, output_file):
    with open(output_file, 'w') as f:
        for phase in phases:
            if phase['type'] != 'brak wdechu':  
                phase_type = "wdech" if phase['type'] == 'wdech' else "wydech"
                f.write(f"{phase_type} {phase['start']} {phase['end']}\n")


def plot_and_save_phases(data, smoothed_data, phases, output_image):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Oryginalny sygnał', alpha=0.5)
    plt.plot(smoothed_data, label='Średnia krocząca', color='orange')

    for phase in phases:
        if phase['type'] == 'wdech':
            plt.axvspan(phase['start'], phase['end'], color='red', alpha=0.3,
                        label='Wdech' if 'Wdech' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif phase['type'] == 'wydech':
            plt.axvspan(phase['start'], phase['end'], color='blue', alpha=0.3,
                        label='Wydech' if 'Wydech' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.legend()
    plt.xlabel('Próbki')
    plt.ylabel('Amplituda')
    plt.title('Analiza Oddechu')
    plt.savefig(output_image)
    plt.close()


def process_audio_file(file_path, window_size, sigma, zero_threshold, breath_threshold_factor):
    rate, data = load_audio(file_path)

    smoothed_data = moving_average(np.abs(data), window_size)
    smoothed_data = smooth_signal(smoothed_data, sigma=sigma)
    smoothed_data[smoothed_data < zero_threshold] = 0
    phases = identify_phases(smoothed_data)
    adjust_phases(phases, breath_threshold_factor)
    filtered_phases = filter_breaths(phases)

    
    processed_phases = keep_longest_inhale_between_exhales(filtered_phases)
    inhale_count = 0  
    exhale_count = 0  
    for phase in phases:
        if phase['type'] == 'wdech':
            inhale_count += 1
        elif phase['type'] == 'wydech':
            exhale_count += 1

    return inhale_count, exhale_count


def main():
    
    window_sizes = range(300, 1001, 50)
    sigmas = np.arange(3.0, 7.5, 0.5)
    zero_thresholds = range(2, 4)
    breath_threshold_factors = np.arange(1.2, 1.41, 0.1)
    
    results = []

    folder_path = r'/one_microphone_recordings'
    save_folder_path = r'/one_microphone_recordings'
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav') and f.startswith('main')]

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        print(f"Przetwarzanie pliku: {file_name}")
        start_time = time.time()
        for ws, sigma, zt, btf in product(window_sizes, sigmas, zero_thresholds, breath_threshold_factors):
            
            inhale_count, exhale_count = process_audio_file(file_path, ws, sigma, zt, btf)

            
            results.append({
                "file_name": file_name,
                "window_size": ws,
                "sigma": sigma,
                "zero_threshold": zt,
                "breath_threshold_factor": btf,
                "inhale_count": inhale_count,
                "exhale_count": exhale_count
            })
        end_time = time.time()
        final_time = end_time - start_time
        print(f"Obróbka jednego pliku: {final_time}")
        output_file = os.path.join(save_folder_path, file_name)
        output_file = output_file.replace(".wav", ".txt")
        with open(output_file, 'w') as f:
            f.write("file_name,window_size,sigma,zero_threshold,breath_threshold_factor,inhale_count,exhale_count\n")
            for result in results:
                f.write(
                    f"{result['file_name']},{result['window_size']},{result['sigma']},{result['zero_threshold']},{result['breath_threshold_factor']},{result['inhale_count']},{result['exhale_count']}\n")
        print(f"Wyniki zapisane do pliku: {output_file}")

if __name__ == "__main__":
    main()
