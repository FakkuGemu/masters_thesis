
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.ndimage import gaussian_filter1d

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


def adjust_phases(phases):
    i = 0
    for phase in phases:
        phase['avg'] = np.mean(phase['values'])
        i = i+1


    wdechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wdech'])

    wydechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wydech'])
    if np.isnan(wdechy_avg):
        wdechy_avg = 0
    elif np.isnan(wydechy_avg):
        wydechy_avg = 0
    threshold = (wdechy_avg + wydechy_avg) / 1.5
    print(f"threshhold: {threshold}")

    for phase in phases:
        if phase['avg'] > threshold:
            phase['type'] = 'wydech'
        else:
            phase['type'] = 'wdech'


def filter_breaths(phases):
    wdechy = [phase for phase in phases if phase['type'] == 'wdech']
    if wdechy:
        wdechy_avg = np.mean([phase['avg'] for phase in wdechy])
        wydechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wydech'])
    else:
        wdechy_avg = 0

    filtered_phases = [phase for phase in phases if
                       phase['type'] == 'wydech' or (phase['type'] == 'wdech' and phase['avg'] >= wdechy_avg)]


    if np.isnan(wdechy_avg):
        wdechy_avg = 0
    elif np.isnan(wydechy_avg):
        wydechy_avg = 0
    threshold = (wdechy_avg + wydechy_avg) / 1.5
    print(f"threshhold: {threshold}")

    for phase in phases:
        if phase['avg'] > threshold:
            phase['type'] = 'wydech'
        else:
            phase['type'] = 'wdech'
    return filtered_phases


def plot_phases(data, smoothed_data, phases, file_name):
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
    plt.savefig(f'C:\\Users\\ivank\\Desktop\\magsiterka\\pythonProject\\oddech\\{file_name}.png')
    plt.show()


def print_phases_info(phases):
    print("Fazy oddechowe:")
    for i, phase in enumerate(phases):
        phase_type = "Wdech" if phase['type'] == 'wdech' else "Wydech"
        print(f"{i + 1}. {phase_type}: Przedział {phase['start']} {phase['end']}")

def load_phases_from_file(file_path):
    phases = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            phase_type = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            phases.append({'type': phase_type, 'start': start, 'end': end})
    return phases

def create_sample_map(phases, total_samples):
    sample_map = np.zeros(total_samples)  

    for phase in phases:
        phase_type = 1 if phase['type'] == 'wdech' else 2
        start, end = phase['start'], phase['end']
        sample_map[start:end] = phase_type

    return sample_map


def calculate_sample_discrepancy(program_phases, file_phases, total_samples):
    program_map = create_sample_map(program_phases, total_samples)
    file_map = create_sample_map(file_phases, total_samples)

    discrepancies = np.sum(program_map != file_map)

    match_percentage = 100 - (discrepancies / total_samples * 100)

    return match_percentage

def main():
    file_name = 'filtr_nos1'
    audio_file_path = f'C:\\Users\\ivank\\Desktop\\magsiterka\\pythonProject\\oddech\\{file_name}.wav'  
    phases_file_path = f'C:\\Users\\ivank\\Desktop\\magsiterka\\pythonProject\\oddech\\{file_name}.txt'  

    rate, data = load_audio(audio_file_path)
    total_samples = len(data)

    window_size = 750
    smoothed_data = moving_average(np.abs(data), window_size)

    smoothed_data = smooth_signal(smoothed_data, sigma=5)
    phases = identify_phases(smoothed_data)

    adjust_phases(phases)

    filtered_phases = filter_breaths(phases)

    print_phases_info(filtered_phases)

    file_phases = load_phases_from_file(phases_file_path)

    for phase in filtered_phases:
        phase['avg'] = np.mean(phase['values'])

    match_percentage = calculate_sample_discrepancy(filtered_phases, file_phases, total_samples)
    print(total_samples)
    print(f"Procent zgodności faz (na podstawie próbek): {match_percentage:.2f}%")

    plot_phases(data, smoothed_data, filtered_phases, file_name)



if __name__ == "__main__":
    main()
