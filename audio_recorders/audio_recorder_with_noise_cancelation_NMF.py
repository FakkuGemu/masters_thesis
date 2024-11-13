import numpy as np
import pyaudio
from scipy.signal import stft, istft
from scipy.io.wavfile import write
from collections import deque
from sklearn.decomposition import NMF



RATE = 44100
CHUNK = 1024
DELAY_SAMPLES = 0  


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

    
    f_1, t_1, Sxx_1 = stft(main_audio, fs=RATE)


    
    Sxx_main = np.abs(Sxx_1)

    
    nmf = NMF(n_components=2, init='random', random_state=0)

    
    W = nmf.fit_transform(Sxx_main)
    H = nmf.components_

    
    Sxx_cleaned = np.dot(W[:, :1], H[:1, :])

    
    _, cleaned_audio = istft(Sxx_cleaned, fs=RATE)

    
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
    save_audio_to_wav("filtered_audio_no_delay.wav", RATE, full_audio_buffer)
    print(f"Saving {len(main_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("main_audio_no_delay.wav", RATE, main_audio_buffer)
    print(f"Saving {len(env_audio_buffer)} samples of audio to file...")
    save_audio_to_wav("env_audio_no_delay.wav", RATE, env_audio_buffer)
    print("Audio saved successfully.")



p = pyaudio.PyAudio()
list_audio_devices()

main_mic_index = 1  
env_mic_index = 0  


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

time.sleep(25)


on_closing()