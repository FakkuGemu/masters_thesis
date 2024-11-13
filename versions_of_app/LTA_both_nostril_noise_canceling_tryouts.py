import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import deque
import tkinter as tk
from scipy.signal import stft, istft
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

CHUNK = 512
RATE = 22050
AMPLITUDE_LIMIT = 30000
full_audio_buffer = []

cumulative_samples = []
timestamps = []
detected_breaths = []
MAX_SAMPLES = 80384
MAX_SAMPLES_CHUNKS = MAX_SAMPLES // CHUNK


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def smooth_signal(data, sigma):
    return gaussian_filter1d(data, sigma=sigma)


def identify_phases(smoothed_data):
    phases = []
    current_phase = None
    for i in range(1, len(smoothed_data)):
        if smoothed_data[i] > 0 and smoothed_data[i - 1] == 0:
            if current_phase is not None:
                phases.append(current_phase)
            current_phase = {'start': i, 'end': None, 'type': 'exhale', 'values': []}

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


def filter_breaths(phases):
    inhales = [phase for phase in phases if phase['type'] == 'inhale']
    if inhales:
        avg_inhales = np.mean([phase['avg'] for phase in inhales])
    else:
        avg_inhales = 0

    filtered_phases = [phase for phase in phases if
                       phase['type'] == 'exhale' or (phase['type'] == 'inhale' and phase['avg'] >= avg_inhales/2)]

    return filtered_phases


def adjust_phases(phases, divider):
    for phase in phases:
        phase['avg'] = np.mean(phase['values'])

    inhales_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'inhale'])
    exhales_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'exhale'])

    if np.isnan(inhales_avg):
        inhales_avg = 0
    if np.isnan(exhales_avg):
        exhales_avg = 0
    threshold = (inhales_avg + exhales_avg) / divider

    for phase in phases:
        if phase['avg'] > threshold:
            phase['type'] = 'exhale'
        else:
            phase['type'] = 'inhale'

def calculate_energy(signal, window_size=4410):
    energy = np.array([np.sum(np.maximum(abs(signal[i:i+window_size]), 0)) for i in range(0, len(signal), window_size)])
    return energy


def calculate_percentage(energy_left, energy_right):
    total_energy = energy_left + energy_right
    percentage_left = (energy_left / total_energy) * 100
    percentage_right = (energy_right / total_energy) * 100
    return percentage_left, percentage_right

def calculate_breath_rate(phases, duration):
    print(f"czas nagrania: {duration}")
    exhale_count = sum(1 for phase in phases if phase['type'] == 'exhale')
    print(f"ilosc wydechów: {exhale_count}")
    breath_rate = (exhale_count / duration) * 60
    return breath_rate

def update_audio_buffer(left_nostril_data, right_nostril_data, env_data):
    left_nostril_audio = np.frombuffer(left_nostril_data, dtype=np.int16)
    right_nostril_audio = np.frombuffer(right_nostril_data, dtype=np.int16)
    env_data = np.frombuffer(env_data, dtype=np.int16)

    left_energy = calculate_energy(left_nostril_audio, len(left_nostril_audio))
    right_energy = calculate_energy(right_nostril_audio, len(right_nostril_audio))
    percentage_left, percentage_right = calculate_percentage(left_energy, right_energy)
    print(f"percentageleft {percentage_left}     percentage right {percentage_right}")
    if left_energy >= right_energy:
        full_audio_buffer.extend(left_nostril_audio)
    else:
        full_audio_buffer.extend(right_nostril_audio)


    if cumulative_samples:
        cumulative_samples.append(cumulative_samples[-1] + CHUNK)
    else:
        cumulative_samples.append(CHUNK)

    timestamps.append(time.time())

    if len(full_audio_buffer) >= RATE * 1:
        process_audio(percentage_left, percentage_right)


def process_audio(percentage_left, percentage_right):
    global cumulative_samples, timestamps
    data = np.array(full_audio_buffer)
    if len(data) > MAX_SAMPLES:
        data = data[-MAX_SAMPLES:]

        cumulative_samples = cumulative_samples[-MAX_SAMPLES_CHUNKS:]
        timestamps = timestamps[-MAX_SAMPLES_CHUNKS:]
    segment_duration = timestamps[-1] - timestamps[0]
    window_size = 500
    smoothed_data = moving_average(np.abs(data), window_size)
    smoothed_data = smooth_signal(smoothed_data, sigma=5)
    threshold_moving_average = 2
    smoothed_data[smoothed_data < threshold_moving_average] = 0
    phases = identify_phases(smoothed_data)
    divider = 1.5
    adjust_phases(phases, divider)
    filtered_phases = filter_breaths(phases)

    breath_rate = calculate_breath_rate(filtered_phases, segment_duration)
    print(f"Breath rate {breath_rate}")
    app.root.after(0, app.update_plot, data, smoothed_data, filtered_phases, breath_rate, percentage_left, percentage_right)



class RealTimeBreathAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Breath Analyzer")
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)

        self.canvas.get_tk_widget().pack(pady=20)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def update_plot(self, data, smoothed_data, phases, breath_rate, percentage_left, percentage_right):
        display_length = 40000
        if len(smoothed_data) > display_length:
            displayed_data = smoothed_data[-display_length:]
            start_index = len(smoothed_data) - display_length
        else:
            displayed_data = smoothed_data
            start_index = 0

        self.ax.clear()
        self.ax.plot(np.arange(start_index, start_index + len(displayed_data)), displayed_data, label='Smoothed Data',
                     color='orange')

        for phase in phases:
            phase_start = phase['start']
            phase_end = phase['end']
            if phase_end >= start_index:
                display_phase_start = max(phase_start, start_index)
                display_phase_end = min(phase_end, start_index + display_length - 1)
                if phase['type'] == 'inhale':
                    self.ax.axvspan(display_phase_start, display_phase_end, color='red', alpha=0.3,
                                    label='Inhale' if 'Inhale' not in plt.gca().get_legend_handles_labels()[1] else "")
                elif phase['type'] == 'exhale':
                    self.ax.axvspan(display_phase_start, display_phase_end, color='blue', alpha=0.3,
                                    label='Exhale' if 'Exhale' not in plt.gca().get_legend_handles_labels()[1] else "")

        self.ax.legend()
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Breath Analysis')
        self.canvas.draw()

    def on_closing(self):
        try:
            if stream_left_nostril.is_active():
                stream_left_nostril.stop_stream()
            stream_left_nostril.close()
            if stream_right_nostril.is_active():
                stream_right_nostril.stop_stream()
            stream_right_nostril.close()
            if stream_env.is_active():
                stream_env.stop_stream()
            stream_env.close()
        except OSError as e:
            print(f"Error stopping or closing the stream: {e}")

        try:
            p.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

        self.root.quit()
        self.root.destroy()


def audio_callback(in_data, frame_count, time_info, status):
    global stream_right_nostril, stream_env
    try:
        right_data = stream_right_nostril.read(frame_count)
        env_data = stream_env.read(frame_count)
        update_audio_buffer(in_data, right_data, env_data)
    except NameError as e:
        print(f"Error: {e}")
    return (in_data, pyaudio.paContinue)


def list_audio_devices():
    device_count = p.get_device_count()
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        print(
            f"Device {i}: {device_info['name']}, Max Input Channels: {device_info['maxInputChannels']}, Default Sample Rate: {device_info['defaultSampleRate']}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeBreathAnalyzerApp(root)

    p = pyaudio.PyAudio()

    list_audio_devices()

    left_mic_index = 1
    right_mic_index = 2
    env_mic_index = 3


    stream_left_nostril = p.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=RATE,
                                 input=True,
                                 frames_per_buffer=CHUNK,
                                 input_device_index=left_mic_index,

                                 stream_callback=audio_callback)

    stream_right_nostril = p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK,
                                  input_device_index=right_mic_index)

    stream_env = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=env_mic_index)



    stream_left_nostril.start_stream()
    stream_right_nostril.start_stream()
    stream_env.start_stream()
    start_time = time.time()
    root.mainloop()

    stream_left_nostril.stop_stream()
    stream_left_nostril.close()
    stream_right_nostril.stop_stream()
    stream_right_nostril.close()
    stream_env.stop_stream()
    stream_env.close()
    p.terminate()