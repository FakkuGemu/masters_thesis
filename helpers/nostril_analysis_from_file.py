import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram, find_peaks


def calculate_energy(signal, window_size=4410):
    energy = np.array([np.sum(np.maximum(abs(signal[i:i+window_size]), 0)) for i in range(0, len(signal), window_size)])
    return energy


def calculate_percentage(energy_left, energy_right):
    total_energy = energy_left + energy_right
    percentage_left = (energy_left / total_energy) * 100
    percentage_right = (energy_right / total_energy) * 100
    return percentage_left, percentage_right


def calculate_breath_rate(signal_length, sample_rate, phases):
    
    recording_duration = signal_length / sample_rate

    
    exhale_count = sum(1 for phase in phases if phase['type'] == 'wydech')

    
    breath_rate = (exhale_count / recording_duration) * 60

    return breath_rate



def plot_spectrogram(fig, ax, signal, sample_rate, title):
    f, t, Sxx = spectrogram(signal, sample_rate)
    img = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    ax.set_ylabel('Częstotliwość [Hz]')
    ax.set_xlabel('Czas [s]')
    ax.set_title(title)
    ax.set_xlim(0, t[-1])  
    fig.colorbar(img, ax=ax, label='Amplituda [dB]')


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


def adjust_phases(phases):
    for phase in phases:
        phase['avg'] = np.mean(phase['values'])

    wdechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wdech'])
    wydechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wydech'])

    if np.isnan(wdechy_avg):
        wdechy_avg = 0
    if np.isnan(wydechy_avg):
        wydechy_avg = 0

    threshold = (wdechy_avg + wydechy_avg) / 1.5

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

def plot_audio_with_phases(left_data, right_data, smoothed_data, left_phases, right_phases):
    plt.figure(figsize=(12, 6))
    plt.plot(left_data, label='Oryginalny sygnał lewa dziurka', alpha=0.5)
    plt.plot(right_data, label='Oryginalny sygnał prawa dziurka', alpha=0.5)
    plt.plot(smoothed_data, label='Średnia krocząca', color='orange')

    for phase in left_phases:
        if phase['type'] == 'wdech':
            plt.axvspan(phase['start'], phase['end'], color='red', alpha=0.3, label='Wdech' if 'Wdech' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif phase['type'] == 'wydech':
            plt.axvspan(phase['start'], phase['end'], color='blue', alpha=0.3, label='Wydech' if 'Wydech' not in plt.gca().get_legend_handles_labels()[1] else "")
    for phase in right_phases:
        if phase['type'] == 'wdech':
            plt.axvspan(phase['start'], phase['end'], color='red', alpha=0.3, label='Wdech' if 'Wdech' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif phase['type'] == 'wydech':
            plt.axvspan(phase['start'], phase['end'], color='blue', alpha=0.3, label='Wydech' if 'Wydech' not in plt.gca().get_legend_handles_labels()[1] else "")


    plt.legend()
    plt.xlabel('Próbki')
    plt.ylabel('Amplituda')
    plt.title('Analiza Oddechu')
    plt.savefig('nozdrza_sygnał.png')
    plt.show()


def plot_breath_analysis(left_signal, right_signal, sample_rate):

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    
    plot_spectrogram(fig, axs[0], left_signal, sample_rate, "Spektrogram - Lewa dziurka")

    
    plot_spectrogram(fig, axs[1], right_signal, sample_rate, "Spektrogram - Prawa dziurka")
    plt.show()

    window_size = int(0.1 * sample_rate)  
    left_energy = calculate_energy(left_signal, window_size)
    right_energy = calculate_energy(right_signal, window_size)
    percentage_left, percentage_right = calculate_percentage(left_energy, right_energy)

    plt.figure(figsize=(12, 6))
    plt.plot(left_energy, label='Energia - Lewa dziurka', color='blue', alpha=0.6)
    plt.plot(right_energy, label='Energia - Prawa dziurka', color='orange', alpha=0.6)
    plt.legend()
    plt.xlabel('Numer okna czasowego')
    plt.ylabel('Ilość energii')
    plt.title('Energia sygnału - obie dziurki')
    plt.savefig('energia.png')
    plt.show()

    print("Procentowy udział lewa dziurka: ", percentage_left)
    print("Procentowy udział prawa dziurka: ", percentage_right)

    
    plt.figure(figsize=(12, 6))
    plt.plot(percentage_left, label='Procent energii - Lewa dziurka', color='blue', alpha=0.6)
    plt.plot(percentage_right, label='Procent energii - Prawa dziurka', color='orange', alpha=0.6)
    plt.legend()
    plt.xlabel('Numer okna czasowego')
    plt.ylabel('Procent energii')
    plt.title('Procent energii sygnału - obie dziurki')
    plt.savefig('procent_energii.png')
    plt.show()

    
    window_size = 1000
    smoothed_data = moving_average(np.abs(left_signal), window_size)
    smoothed_data = smooth_signal(smoothed_data, sigma=10)
    threshold_moving_average = 5
    smoothed_data[smoothed_data < threshold_moving_average] = 0
    phases = identify_phases(smoothed_data)
    adjust_phases(phases)
    filtered_phases = filter_breaths(phases)
    processed_phases = keep_longest_inhale_between_exhales(filtered_phases)

    
    smoothed_data2 = moving_average(np.abs(right_signal), window_size)
    smoothed_data2 = smooth_signal(smoothed_data2, sigma=10)
    smoothed_data2[smoothed_data2 < threshold_moving_average] = 0
    phases2 = identify_phases(smoothed_data2)
    adjust_phases(phases2)
    filtered_phases2 = filter_breaths(phases2)
    processed_phases2 = keep_longest_inhale_between_exhales(filtered_phases2)

    plot_audio_with_phases(left_signal, right_signal, smoothed_data, processed_phases, processed_phases2)
    breath_rate = calculate_breath_rate(len(left_signal), sample_rate, processed_phases2)
    print(f"Breath rate {breath_rate}")


name_of_file="13"
name_of_file_base="nostril_audio\\left_nostril.wav"
name_of_file_left=name_of_file_base.replace("left_nostril","left_nostril"+name_of_file)
name_of_file_right = name_of_file_base.replace("left_nostril","right_nostril"+name_of_file)
sample_rate, left_signal = wavfile.read(name_of_file_left)
_, right_signal = wavfile.read(name_of_file_right)




plot_breath_analysis(left_signal, right_signal, sample_rate)