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



def audio_callback(in_data, frame_count, time_info, status, stream_env):
    
    main_audio = np.frombuffer(in_data, dtype=np.int16)
    main_audio_buffer.extend(main_audio.astype(np.int16))  

    
    env_audio = np.frombuffer(stream_env.read(frame_count), dtype=np.int16)
    env_audio_buffer.extend(env_audio.astype(np.int16))
    
    f_1, t_1, Sxx_1 = stft(main_audio, fs=RATE)
    f_2, t_2, Sxx_2 = stft(env_audio, fs=RATE)

    magnitude_1 = np.abs(Sxx_1)
    magnitude_2 = np.abs(Sxx_2)
    phase_1 = np.angle(Sxx_1)


    magnitude_cleaned = np.maximum(magnitude_1 - magnitude_2, 0)


    Zxx_cleaned = magnitude_cleaned * np.exp(1j * phase_1)


    _, cleaned_audio = istft(Zxx_cleaned, fs=RATE)
    
    full_audio_buffer.extend(cleaned_audio.astype(np.int16))  

    return (in_data, pyaudio.paContinue)



def on_closing():
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

    
    print(f"Saving {len(full_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("noise_cancelation_files\\full_audio_no_delay_threshold,04_dB,60.wav", RATE, full_audio_buffer)
    print(f"Saving {len(main_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("../noise_cancelation_files/main_audio_no_delay_threshold,04_dB,60.wav", RATE, main_audio_buffer)
    print(f"Saving {len(env_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("../noise_cancelation_files/env_audio_no_delay_threshold,04_dB,60.wav", RATE, env_audio_buffer)
    print("Audio saved successfully.")



p = pyaudio.PyAudio()
list_audio_devices()

main_mic_index = 1  
env_mic_index = 2  


stream_env = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=env_mic_index)


stream_main = p.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK,
                     input_device_index=main_mic_index,
                     stream_callback=lambda in_data, frame_count, time_info, status: audio_callback(in_data,
                                                                                                    frame_count,
                                                                                                    time_info, status,
                                                                                                    stream_env))


stream_main.start_stream()
stream_env.start_stream()


import time

time.sleep(20)


on_closing()