import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import deque
import tkinter as tk
from scipy.signal import stft, istft
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


RATE = 44100
CHUNK = 1024
WINDOW_SIZE = 500
SIGMA = 10
AMPLITUDE_LIMIT = 30000
DELAY_SAMPLES = 10000  # Liczba próbek opóźnienia


main_audio_fifo = deque(maxlen=DELAY_SAMPLES + CHUNK)

main_audio_buffer = deque(maxlen=10 * RATE)
env_audio_buffer = deque(maxlen=10 * RATE)
full_audio_buffer = []

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
                       phase['type'] == 'exhale' or (phase['type'] == 'inhale' and phase['avg'] >= avg_inhales)]

    return filtered_phases


def adjust_phases(phases):
    for phase in phases:
        phase['avg'] = np.mean(phase['values'])

    inhales_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'inhale'])
    exhales_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'exhale'])

    if np.isnan(inhales_avg):
        inhales_avg = 0
    if np.isnan(exhales_avg):
        exhales_avg = 0
    threshold = (inhales_avg + exhales_avg) / 1.5

    for phase in phases:
        if phase['avg'] > threshold:
            phase['type'] = 'exhale'
        else:
            phase['type'] = 'inhale'


def update_audio_buffer(in_data, env_data):
    global main_audio_fifo


    main_audio = np.frombuffer(in_data, dtype=np.int16)
    env_audio = np.frombuffer(env_data, dtype=np.int16)

    main_audio_fifo.extend(main_audio)


    if len(main_audio_fifo) >= DELAY_SAMPLES:

        delayed_main_audio = np.array(list(main_audio_fifo)[:len(main_audio)])
    else:
        delayed_main_audio = np.zeros(len(main_audio))


    f_1, t_1, Sxx_1 = stft(delayed_main_audio, fs=RATE)
    f_2, t_2, Sxx_2 = stft(env_audio, fs=RATE)


    magnitude_1 = np.abs(Sxx_1)
    magnitude_2 = np.abs(Sxx_2)
    phase_1 = np.angle(Sxx_1)


    magnitude_cleaned = np.maximum(magnitude_1 - magnitude_2, 0)


    Zxx_cleaned = magnitude_cleaned * np.exp(1j * phase_1)


    _, cleaned_audio = istft(Zxx_cleaned, fs=RATE)

    full_audio_buffer.extend(cleaned_audio)
    main_audio_buffer.extend(cleaned_audio)
    env_audio_buffer.extend(env_audio)


    if len(full_audio_buffer) >= RATE * 1:
        process_audio()


def process_audio():
    MAX_SAMPLES = 200000
    data = np.array(full_audio_buffer)
    if len(data) > MAX_SAMPLES:
        data = data[-MAX_SAMPLES:]

    phases_data = moving_average(np.abs(data), WINDOW_SIZE)
    phases_data[phases_data < 2] = 0

    phases_data = smooth_signal(phases_data, sigma=10)
    phases = identify_phases(phases_data)

    adjust_phases(phases)
    filtered_phases = filter_breaths(phases)

    app.root.after(0, app.update_plot, data, phases_data, filtered_phases)


class RealTimeBreathAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Breath Analyzer")
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=20)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_plot(self, data, smoothed_data, phases):
        display_length = 80000
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
            if stream_main.is_active():
                stream_main.stop_stream()
            stream_main.close()
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
    env_data = stream_env.read(frame_count)
    update_audio_buffer(in_data, env_data)
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

    main_mic_index = int(input("Enter the index of the main microphone: "))
    noise_mic_index = int(input("Enter the index of the noise microphone: "))


    stream_main = p.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=RATE,
                         input=True,
                         frames_per_buffer=CHUNK,
                         input_device_index=main_mic_index,
                         stream_callback=audio_callback)

    stream_env = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        input_device_index=noise_mic_index,
                        frames_per_buffer=CHUNK)

    stream_main.start_stream()
    stream_env.start_stream()

    root.mainloop()

    stream_main.stop_stream()
    stream_main.close()
    stream_env.stop_stream()
    stream_env.close()
    p.terminate()