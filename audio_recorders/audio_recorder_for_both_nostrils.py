import numpy as np
import pyaudio
from scipy.signal import stft, istft
from scipy.io.wavfile import write
from collections import deque


RATE = 44100
CHUNK = 1024



full_audio_buffer = []
left_nostril_audio_buffer = []
right_nostril_audio_buffer = []

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
    left_nostril_audio_buffer.extend(main_audio.astype(np.int16))  

    
    env_audio = np.frombuffer(stream_env.read(frame_count), dtype=np.int16)
    right_nostril_audio_buffer.extend(env_audio.astype(np.int16))

    return (in_data, pyaudio.paContinue)



def on_closing():
    try:
        if stream_left_nostril.is_active():
            stream_left_nostril.stop_stream()
        stream_left_nostril.close()
        if stream_right_nostril.is_active():
            stream_right_nostril.stop_stream()
        stream_right_nostril.close()
    except OSError as e:
        print(f"Error stopping or closing the stream: {e}")

    try:
        p.terminate()
    except Exception as e:
        print(f"Error terminating PyAudio: {e}")

    
    print(f"Saving {len(left_nostril_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("../noise_cancelation_all_methods/main_audio6.wav", RATE, left_nostril_audio_buffer)
    print(f"Saving {len(right_nostril_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("../noise_cancelation_all_methods/env_audio6.wav", RATE, right_nostril_audio_buffer)




p = pyaudio.PyAudio()
list_audio_devices()

left_mic_index = 1  
right_mic_index = 2


stream_right_nostril = p.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK,
                              input_device_index=right_mic_index)


stream_left_nostril = p.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=RATE,
                             input=True,
                             frames_per_buffer=CHUNK,
                             input_device_index=left_mic_index,
                             stream_callback=lambda in_data, frame_count, time_info, status: audio_callback(in_data,
                                                                                                    frame_count,
                                                                                                    time_info, status,
                                                                                                    stream_right_nostril))


stream_left_nostril.start_stream()
stream_right_nostril.start_stream()


import time

time.sleep(20)


on_closing()