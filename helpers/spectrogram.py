import librosa
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft



CHUNK = 4096  
FORMAT = pyaudio.paInt16  
CHANNELS = 1  
RATE = 44100  
LOW_FREQ_CUTOFF = 2000  


p = pyaudio.PyAudio()



main_mic_sensitivity = 0.3  
noise_mic_sensitivity = 1.0  




def initialize_stream(device_index):
   return p.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 input_device_index=device_index,
                 frames_per_buffer=CHUNK)




def list_audio_devices():
   device_count = p.get_device_count()
   for i in range(device_count):
       device_info = p.get_device_info_by_index(i)
       print(
           f"Device {i}: {device_info['name']}, Max Input Channels: {device_info['maxInputChannels']}, Default Sample Rate: {device_info['defaultSampleRate']}")




list_audio_devices()


main_mic_index = int(input("Enter the index of the main microphone: "))
noise_mic_index = int(input("Enter the index of the noise microphone: "))


main_stream = initialize_stream(main_mic_index)
noise_stream = initialize_stream(noise_mic_index)


plt.ion()
fig, (ax_raw, ax_noise, ax_spectrogram, ax_final_signal) = plt.subplots(4, 1)



x = np.arange(0, 2 * CHUNK, 2)
line_raw, = ax_raw.plot(x, np.random.rand(CHUNK), '-', lw=2)
ax_raw.set_ylim([-32768, 32767])  
ax_raw.set_title("Live Raw Audio Signal")
ax_raw.set_xlabel("Samples")
ax_raw.set_ylabel("Amplitude")


x = np.arange(0, 2 * CHUNK, 2)
line_noise, = ax_noise.plot(x, np.random.rand(CHUNK), '-', lw=2)
ax_noise.set_ylim([-32768, 32767])  
ax_noise.set_title("Live Raw Audio Signal")
ax_noise.set_xlabel("Samples")
ax_noise.set_ylabel("Amplitude")



ax_spectrogram.set_title("Live Audio Spectrogram")
ax_spectrogram.set_xlabel("Time")
ax_spectrogram.set_ylabel("Frequency [Hz]")



line_final_signal, = ax_final_signal.plot(x, np.random.rand(CHUNK), '-', lw=2)
ax_final_signal.set_ylim([-32768, 32767])  
ax_final_signal.set_title("Final Audio Signal (after Spectrogram Inversion)")
ax_final_signal.set_xlabel("Samples")
ax_final_signal.set_ylabel("Amplitude")




def remove_internal_noise(signal, threshold=(-2, 2)):
   """Usuwa szumy wewnętrzne z zakresu -2 do 2."""
   signal_cleaned = np.where((signal > threshold[0]) & (signal < threshold[1]), 0, signal)
   return signal_cleaned




def update_raw_signal_plot(signal):
   line_raw.set_ydata(signal)
   fig.canvas.draw()
   fig.canvas.flush_events()

def update_noise_signal_plot(signal):
   line_noise.set_ydata(signal)
   fig.canvas.draw()
   fig.canvas.flush_events()


def update_spectrogram_plot(signal, noise_signal, rate):
   """Aktualizuje spektrogram po odjęciu sygnału z mikrofonu szumów od sygnału głównego."""
   
   f_main, t_main, Sxx_main = stft(signal, fs=rate, nperseg=256)
   f_noise, t_noise, Sxx_noise = stft(noise_signal, fs=rate, nperseg=256)

   

   threshold = 0.01
   amplitude_main = np.abs(Sxx_main)
   amplitude_noise = np.abs(Sxx_noise)
   mask = (amplitude_main > threshold) & (amplitude_noise > threshold)
   Sxx_filtered = Sxx_main.copy()
   Sxx_filtered[mask] = Sxx_main[mask] - Sxx_noise[mask]

   
   S_diff_db = librosa.amplitude_to_db(np.abs(Sxx_filtered),ref=np.max)
   ax_spectrogram.clear()
   ax_spectrogram.pcolormesh(t_main, f_main, S_diff_db, shading='gouraud')
   ax_spectrogram.set_ylim([0, LOW_FREQ_CUTOFF])  
   ax_spectrogram.set_title("Live Audio Spectrogram (Noise Reduction Applied)")
   ax_spectrogram.set_xlabel("Time")
   ax_spectrogram.set_ylabel("Frequency [Hz]")
   plt.pause(0.01)


   
   _, final_signal = istft(Sxx_filtered, fs=rate)
   return final_signal




def update_final_signal_plot(signal):
   line_final_signal.set_ydata(signal[:CHUNK])  
   fig.canvas.draw()
   fig.canvas.flush_events()




try:
   while True:
       
       main_data = main_stream.read(CHUNK)
       noise_data = noise_stream.read(CHUNK)


       
       main_signal = np.frombuffer(main_data, dtype=np.int16)
       noise_signal = np.frombuffer(noise_data, dtype=np.int16)


       main_signal = main_signal * main_mic_sensitivity
       noise_signal = noise_signal * noise_mic_sensitivity


       main_signal_cleaned = remove_internal_noise(main_signal)
       noise_signal_cleaned = remove_internal_noise(noise_signal)


       update_raw_signal_plot(main_signal_cleaned)

       update_noise_signal_plot(noise_signal_cleaned)


       final_signal = update_spectrogram_plot(main_signal_cleaned, noise_signal_cleaned, RATE)


       update_final_signal_plot(final_signal)


except KeyboardInterrupt:
   print("Stopping...")


finally:
   main_stream.stop_stream()
   main_stream.close()
   noise_stream.stop_stream()
   noise_stream.close()
   p.terminate()
