import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import ttk
from scipy.signal import stft, istft
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from collections import deque

RATE = 22050
CHUNK = 512
AMPLITUDE_LIMIT = 30000
BUFFER_LIMIT = RATE * 2  
MAX_SAMPLES = 80384
MAX_SAMPLES_CHUNKS = MAX_SAMPLES // CHUNK  

full_audio_buffer = deque(maxlen=MAX_SAMPLES)
timestamps = deque(maxlen=MAX_SAMPLES // CHUNK)

detected_breaths = []



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
    exhale_count = sum(1 for phase in phases if phase['type'] == 'exhale')
    breath_rate = (exhale_count / duration) * 60
    return breath_rate

def update_audio_buffer(left_nostril_data, right_nostril_data, env_data):
    left_nostril_audio = np.frombuffer(left_nostril_data, dtype=np.int16)
    right_nostril_audio = np.frombuffer(right_nostril_data, dtype=np.int16)
    env_audio = np.frombuffer(env_data, dtype=np.int16)

    left_energy = calculate_energy(left_nostril_audio, len(left_nostril_audio))
    right_energy = calculate_energy(right_nostril_audio, len(right_nostril_audio))
    percentage_left, percentage_right = calculate_percentage(left_energy, right_energy)
    if left_energy >= right_energy:
        temp_main_audio = left_nostril_audio
    else:
        temp_main_audio = right_nostril_audio

    f_1, t_1, Sxx_1 = stft(temp_main_audio, fs=RATE)
    f_2, t_2, Sxx_2 = stft(env_audio, fs=RATE)

    magnitude_1 = np.abs(Sxx_1)
    magnitude_2 = np.abs(Sxx_2)
    phase_1 = np.angle(Sxx_1)

    magnitude_cleaned = np.maximum(magnitude_1 - magnitude_2, 0)

    Zxx_cleaned = magnitude_cleaned * np.exp(1j * phase_1)
    _, cleaned_audio = istft(Zxx_cleaned, fs=RATE)
    full_audio_buffer.extend(cleaned_audio)

    timestamps.append(time.time())
    if len(full_audio_buffer) >= RATE * 1:
        process_audio(percentage_left, percentage_right)


def process_audio(percentage_left, percentage_right):
    global timestamps
    data = np.array(full_audio_buffer)
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
    app.root.after(0, app.update_plot, full_audio_buffer, smoothed_data, filtered_phases, breath_rate, percentage_left, percentage_right)


class MicrophoneSelectionWindow(tk.Toplevel):
    def __init__(self, parent, devices):
        super().__init__(parent)
        self.title("Select Microphones")
        self.geometry("600x300")
        self.devices = devices

        
        tk.Label(self, text="Left Nostril Microphone:").pack(pady=5)
        self.left_mic_combo = ttk.Combobox(self, values=self.devices, state="readonly", width=50)
        self.left_mic_combo.pack()
        self.left_mic_label = tk.Label(self, text="")
        self.left_mic_label.pack()
        self.left_mic_combo.bind("<<ComboboxSelected>>", self.update_left_label)

        tk.Label(self, text="Right Nostril Microphone:").pack(pady=5)
        self.right_mic_combo = ttk.Combobox(self, values=self.devices, state="readonly", width=50)
        self.right_mic_combo.pack()
        self.right_mic_label = tk.Label(self, text="")
        self.right_mic_label.pack()
        self.right_mic_combo.bind("<<ComboboxSelected>>", self.update_right_label)

        tk.Label(self, text="Environment Microphone:").pack(pady=5)
        self.env_mic_combo = ttk.Combobox(self, values=self.devices, state="readonly", width=50)
        self.env_mic_combo.pack()
        self.env_mic_label = tk.Label(self, text="")
        self.env_mic_label.pack()
        self.env_mic_combo.bind("<<ComboboxSelected>>", self.update_env_label)

        
        self.confirm_button = tk.Button(self, text="Confirm Selection", command=self.on_confirm)
        self.confirm_button.pack(pady=10)

        self.selected_mics = None

    def update_left_label(self, event):
        self.left_mic_label.config(text=f"Selected: {self.left_mic_combo.get()}")

    def update_right_label(self, event):
        self.right_mic_label.config(text=f"Selected: {self.right_mic_combo.get()}")

    def update_env_label(self, event):
        self.env_mic_label.config(text=f"Selected: {self.env_mic_combo.get()}")

    def on_confirm(self):
        
        self.selected_mics = (
            self.left_mic_combo.current(),
            self.right_mic_combo.current(),
            self.env_mic_combo.current()
        )
        self.destroy()

    def get_selected_mics(self):
        return self.selected_mics


class RealTimeBreathAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Breath Analyzer")

        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=20)

        
        self.breath_rate_label = tk.Label(root, text="Breathing Rate: 0.0 breaths/min")
        self.breath_rate_label.pack(pady=5)

        self.dominance_label = tk.Label(root, text="Dominant Nostril: N/A")
        self.dominance_label.pack(pady=5)

        self.p = pyaudio.PyAudio()

        
        devices = self.list_audio_devices()
        self.mic_selection_window = MicrophoneSelectionWindow(self.root, devices)
        self.root.wait_window(self.mic_selection_window)

        
        self.selected_mics = self.mic_selection_window.get_selected_mics()
        if self.selected_mics:
            self.left_mic_index, self.right_mic_index, self.env_mic_index = self.selected_mics
            self.setup_audio_streams()

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

        
        self.breath_rate_label.config(text=f"Breathing Rate: {breath_rate:.1f} breaths/min")

        
        if abs(percentage_left - percentage_right) <= 20:
            dominant_nostril = "Balanced"
        else:
            dominant_nostril = "Left" if percentage_left > percentage_right else "Right"

        self.dominance_label.config(text=f"Dominant Nostril: {dominant_nostril}")

        self.ax.legend()
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Breath Analysis')
        self.canvas.draw()

    def on_closing(self):
        
        try:
            if self.stream_left_nostril.is_active():
                self.stream_left_nostril.stop_stream()
            self.stream_left_nostril.close()
            if self.stream_right_nostril.is_active():
                self.stream_right_nostril.stop_stream()
            self.stream_right_nostril.close()
            if self.stream_env.is_active():
                self.stream_env.stop_stream()
            self.stream_env.close()
        except Exception as e:
            print(f"Error closing streams: {e}")

        self.p.terminate()
        self.root.quit()
        self.root.destroy()

    def list_audio_devices(self):
        device_count = self.p.get_device_count()
        devices = []
        for i in range(device_count):
            device_info = self.p.get_device_info_by_index(i)
            devices.append(f"{device_info['name']}")
        return devices

    def audio_callback(self, in_data, frame_count, time_info, status):
        right_data = self.stream_right_nostril.read(frame_count)
        env_data = self.stream_env.read(frame_count)
        update_audio_buffer(in_data, right_data, env_data)
        return (in_data, pyaudio.paContinue)

    def setup_audio_streams(self):
        
        self.stream_left_nostril = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=self.left_mic_index,
            stream_callback=self.audio_callback
        )
        self.stream_right_nostril = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=self.right_mic_index
        )
        self.stream_env = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=self.env_mic_index
        )
        self.stream_left_nostril.start_stream()
        self.stream_right_nostril.start_stream()
        self.stream_env.start_stream()
        self.start_time = time.time()


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeBreathAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()