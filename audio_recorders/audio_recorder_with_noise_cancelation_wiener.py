import numpy as np
import pyaudio
from joblib import delayed
from scipy.signal import stft, istft, wiener
from scipy.io.wavfile import write
from collections import deque


RATE = 44100
CHUNK = 1024
DELAY_SAMPLES = 10240  


main_audio_fifo = deque(maxlen=DELAY_SAMPLES)
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

    
    main_audio_fifo.extend(main_audio)

    
    if len(main_audio_fifo) >= DELAY_SAMPLES:
        
        delayed_main_audio = np.array(list(main_audio_fifo)[:len(main_audio)])
    else:
        delayed_main_audio = np.zeros(len(main_audio))  

    
    f_1, t_1, Sxx_1 = stft(delayed_main_audio, fs=RATE)

    cleaned_spectrogram = wiener(Sxx_1)

    
    _, cleaned_audio = istft(cleaned_spectrogram, fs=RATE)

    
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
    save_audio_to_wav("filtered_audio_weiner1.wav", RATE, full_audio_buffer)
    print(f"Saving {len(main_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("main_audio_weiner1.wav", RATE, main_audio_buffer)
    print(f"Saving {len(env_audio_buffer)} samples of audio to file...")
    
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