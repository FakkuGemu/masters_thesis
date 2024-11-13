import numpy as np
import pyaudio
from scipy.signal import stft, istft
from scipy.io.wavfile import write
from collections import deque

RATE = 44100
CHUNK = 1024

full_audio_buffer = []
main_audio_buffer = []
env_audio_buffer = []


def list_audio_devices():
    device_count = p.get_device_count()
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        print(
            f"Device {i}: {device_info['name']}, Max Input Channels: {device_info['maxInputChannels']}, Default Sample Rate: {device_info['defaultSampleRate']}")


def save_audio_to_wav(filename, rate, data):
    data = np.array(data, dtype=np.int16)
    write(filename, rate, data)


def audio_callback(in_data, frame_count, time_info, status):
    main_audio = np.frombuffer(in_data, dtype=np.int16)
    full_audio_buffer.extend(main_audio.astype(np.int16))
    return (in_data, pyaudio.paContinue)


def on_closing():
    try:
        if stream_main.is_active():
            stream_main.stop_stream()
        stream_main.close()
    except OSError as e:
        print(f"Error stopping or closing the stream: {e}")

    try:
        p.terminate()
    except Exception as e:
        print(f"Error terminating PyAudio: {e}")

    print(f"Saving {len(full_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("one_microphone_recordings\\inhale56.wav", RATE, full_audio_buffer)



p = pyaudio.PyAudio()


main_mic_index = 1



stream_main = p.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK,
                     input_device_index=main_mic_index,
                     stream_callback=lambda in_data, frame_count, time_info, status: audio_callback(in_data,
                                                                                                    frame_count,
                                                                                                    time_info, status))

stream_main.start_stream()


import time

time.sleep(2)

on_closing()