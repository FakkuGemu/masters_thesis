import numpy as np
import wave
import struct
import matplotlib.pyplot as plt

def generate_breath_signal(samples, amplitude, phase_type='inhale'):
    """Generuj sygnał dla wdechu lub wydechu z odpowiednią obwiednią."""
    t = np.linspace(0, 1, samples)

    
    base_signal = np.random.normal(0, 1, samples)

    
    if phase_type == 'inhale':
        envelope = np.where(t < 0.7, t / 0.7, np.exp(-3 * (t - 0.7)))
    elif phase_type == 'exhale':
        envelope = np.where(t < 0.3, t / 0.3, 1 - (t - 0.3) / 0.7)

    
    signal = amplitude * base_signal * envelope
    return signal


def smooth_signal_every_50(signal):
    """Wygładzanie sygnału poprzez wybór co 50 próbki i interpolację pomiędzy nimi."""
    smoothed_signal = np.copy(signal)  

    
    total_samples = len(signal)

    for i in range(0, total_samples - 50, 50):
        
        start_value = signal[i]
        end_value = signal[i + 50]

        
        for j in range(1, 50):
            
            smoothed_signal[i + j] = start_value + (j / 50) * (end_value - start_value)

    return smoothed_signal


def generate_breath_data(total_samples):
    """Generuje dane oddechowe zgodnie z podanym schematem."""
    data = []
    current_sample = 0

    while current_sample < total_samples:
        
        wdech_upper = np.random.randint(400, 600)
        wdech_length = np.random.randint(15000, 72200)
        wdech_signal = generate_breath_signal(wdech_length, wdech_upper / 1000, 'inhale')  

        
        wdech_signal = smooth_signal_every_50(wdech_signal)

        
        data.extend(wdech_signal)
        current_sample += wdech_length

        if current_sample >= total_samples:
            break

        
        empty_breath_length = np.random.randint(900, 28000)
        empty_breath_signal = np.zeros(empty_breath_length)

        
        data.extend(empty_breath_signal)
        current_sample += empty_breath_length

        if current_sample >= total_samples:
            break

        
        wydech_upper = np.random.randint(1500, 2850)
        wydech_length = np.random.randint(113085, 155000)
        wydech_signal = generate_breath_signal(wydech_length, wydech_upper / 1000, 'exhale')  

        
        wydech_signal = smooth_signal_every_50(wydech_signal)

        
        data.extend(wydech_signal)
        current_sample += wydech_length

        if current_sample >= total_samples:
            break

        
        post_wydech_length = np.random.randint(22000, 99000)
        post_wydech_signal = np.zeros(post_wydech_length)

        
        data.extend(post_wydech_signal)
        current_sample += post_wydech_length

    
    data = np.array(data[:total_samples])

    return data


def save_to_wav(file_name, data, sample_rate=44100):
    """Zapisz wygenerowany sygnał do pliku WAV."""
    wav_file = wave.open(file_name, 'w')
    n_channels = 1  
    sampwidth = 2  
    n_frames = len(data)

    
    wav_file.setparams((n_channels, sampwidth, sample_rate, n_frames, 'NONE', 'not compressed'))

    
    for sample in data:
        sample = int(sample * 32767)  
        sample = max(-32768, min(32767, sample))  

        
        wav_file.writeframes(struct.pack('h', sample))

    wav_file.close()


def main():
    
    total_samples = np.random.randint(1_000_000, 1_000_001)  
    output_file = "wygenerowane_dane_oddechowe_smooth.wav"  


    
    breath_data = generate_breath_data(total_samples)
    plt.plot(breath_data, label='Oryginalny sygnał', alpha=0.5)
    plt.show()
    
    save_to_wav(output_file, breath_data)

    print(f"Zapisano wygenerowane dane oddechowe do {output_file} ({len(breath_data)} próbek)")


if __name__ == "__main__":
    main()
