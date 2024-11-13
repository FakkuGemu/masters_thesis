import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import deque
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


RATE = 16000
CHUNK = 1024
WINDOW_SIZE = 2000
SIGMA = 10
AMPLITUDE_LIMIT = 30000


full_audio_buffer = []
audio_buffer = deque(maxlen=10 * RATE)


detected_breaths = []

def moving_average(data, window_size):
    filtred_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    return filtred_data


def smooth_signal(data, sigma):
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

def filter_breaths(phases):

    wdechy = [phase for phase in phases if phase['type'] == 'wdech']
    if wdechy:
        avg_wdechy = np.mean([phase['avg'] for phase in wdechy])
    else:
        avg_wdechy = 0


    filtered_phases = [phase for phase in phases if
                       phase['type'] == 'wydech' or (phase['type'] == 'wdech' and phase['avg'] >= avg_wdechy)]

    return filtered_phases

def adjust_phases(phases):


    for phase in phases:
        phase['avg'] = np.mean(phase['values'])



    wdechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wdech'])

    wydechy_avg = np.mean([phase['avg'] for phase in phases if phase['type'] == 'wydech'])

    if np.isnan(wdechy_avg):
        wdechy_avg = 0
    elif np.isnan(wydechy_avg):
        wydechy_avg = 0
    threshold = (wdechy_avg + wydechy_avg) / 2


    for phase in phases:
        if phase['avg'] > threshold:
            phase['type'] = 'wydech'
        else:
            phase['type'] = 'wdech'


def update_audio_buffer(in_data):

    audio_data = np.frombuffer(in_data, dtype=np.int16)

    audio_data = np.clip(np.abs(audio_data), 0, AMPLITUDE_LIMIT)

    full_audio_buffer.extend(audio_data)
    audio_buffer.extend(audio_data)

    if len(full_audio_buffer) >= RATE * 1:
        process_audio()


def process_audio():
    MAX_SAMPLES = 200000

    data = np.array(full_audio_buffer)
    if len(data) > MAX_SAMPLES:
        data = data[-MAX_SAMPLES:]

    phases_data = moving_average(np.abs(data), WINDOW_SIZE)

    phases_data[phases_data < 1] = 0

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

    def update_plot(self, data, smoothed_data,  phases):

        display_length = 80000



        if len(smoothed_data) > display_length:
            displayed_data = smoothed_data[-display_length:]
            start_index = len(smoothed_data) - display_length
        else:
            displayed_data = smoothed_data
            start_index = 0

        self.ax.clear()

        self.ax.plot(np.arange(start_index, start_index + len(displayed_data)), displayed_data,
                     label='Średnia krocząca', color='orange')

        for phase in phases:
            phase_start = phase['start']
            phase_end = phase['end']


            if phase_end >= start_index:

                display_phase_start = max(phase_start, start_index)
                display_phase_end = min(phase_end, start_index + display_length - 1)


                if phase['type'] == 'wdech':
                    self.ax.axvspan(display_phase_start, display_phase_end, color='red', alpha=0.3,
                                    label='Wdech' if 'Wdech' not in plt.gca().get_legend_handles_labels()[1] else "")
                elif phase['type'] == 'wydech':
                    self.ax.axvspan(display_phase_start, display_phase_end, color='blue', alpha=0.3,
                                    label='Wydech' if 'Wydech' not in plt.gca().get_legend_handles_labels()[1] else "")

        self.ax.legend()
        self.ax.set_xlabel('Próbki')
        self.ax.set_ylabel('Amplituda')
        self.ax.set_title('Analiza Oddechu')


        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
    def safe_plot(self):
        data = np.array(full_audio_buffer)


        phases_data = moving_average(np.abs(data), WINDOW_SIZE)

        phases_data = smooth_signal(phases_data, sigma=10)
        phases = identify_phases(phases_data)
        adjust_phases(phases)
        filtered_phases = filter_breaths(phases)
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Oryginalny sygnał', alpha=0.5)
        plt.plot(phases_data, label='Średnia krocząca', color='orange')

        for phase in filtered_phases:
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
        plt.show()
        plt.savefig("wykres")

    def on_closing(self):

        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except OSError as e:
            print(f"Error stopping or closing the stream: {e}")

        try:
            p.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

        self.safe_plot()
        self.root.quit()
        self.root.destroy()





def audio_callback(in_data, frame_count, time_info, status):
    update_audio_buffer(in_data)
    return (in_data, pyaudio.paContinue)


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeBreathAnalyzerApp(root)


    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)


    stream.start_stream()


    root.mainloop()


    stream.stop_stream()
    stream.close()
    p.terminate()

