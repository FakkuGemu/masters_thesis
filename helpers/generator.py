import numpy as np
import wave
import struct
import matplotlib.pyplot as plt


def generate_breath_signal(samples, amplitude, phase_type='inhale'):
    t = np.linspace(0, 1, samples)

    base_signal = np.random.normal(0, 1, samples)

    window_size = 100
    base_signal = np.convolve(base_signal, np.ones(window_size) / window_size, mode='same')

    if phase_type == 'inhale':
        envelope = np.where(t < 0.7, t / 0.7, np.exp(-3 * (t - 0.7)))
    elif phase_type == 'exhale':
        envelope = np.where(t < 0.3, t / 0.3, 1 - (t - 0.3) / 0.7)

    envelope /= np.max(envelope)

    signal = base_signal * envelope

    signal = amplitude * signal / np.max(np.abs(signal))

    return signal



breath_events = []
def generate_random_value(min_value, max_value):
    mean = (min_value + max_value)/2
    std_dev = (max_value - mean)/2
    value = np.random.normal(mean, std_dev)
    return int(np.clip(np.round(value), min_value, max_value))


def generate_breath_data(total_samples, file_name="breath_data"):
    data = []
    current_sample = 0
    breath_events = []
    file_name = file_name + ".txt"

    with open(file_name, "w") as f:
        while current_sample < total_samples:
            
            wdech_upper = generate_random_value(min_value=400, max_value=600)
            wdech_length = generate_random_value(min_value=15000, max_value=72200)
            wdech_signal = generate_breath_signal(wdech_length, wdech_upper, 'inhale')  

            
            breath_events.append(f"wdech {current_sample} {current_sample + wdech_length}")
            f.write(f"wdech {current_sample} {current_sample + wdech_length}\n")

            
            data.extend(wdech_signal)
            current_sample += wdech_length

            if current_sample >= total_samples:
                break

            
            empty_breath_length = generate_random_value(min_value=4000, max_value=28000)
            empty_breath_signal = np.zeros(empty_breath_length)

            
            data.extend(empty_breath_signal)
            current_sample += empty_breath_length

            if current_sample >= total_samples:
                break

            
            wydech_upper = generate_random_value(min_value=1500, max_value=2850)
            wydech_length = generate_random_value(min_value=113085, max_value=155000)
            wydech_signal = generate_breath_signal(wydech_length, wydech_upper, 'exhale')  

            
            breath_events.append(f"wydech {current_sample} {current_sample + wydech_length}")
            f.write(f"wydech {current_sample} {current_sample + wydech_length}\n")

            
            data.extend(wydech_signal)
            current_sample += wydech_length

            if current_sample >= total_samples:
                break

            
            post_wydech_length = generate_random_value(min_value=22000, max_value=99000)
            post_wydech_signal = np.zeros(post_wydech_length)

            
            data.extend(post_wydech_signal)
            current_sample += post_wydech_length

        
        data = np.array(data[:total_samples])

    return data

def save_to_wav(file_name, data, sample_rate=44100):
    file_name = file_name + ".wav"
    wav_file = wave.open(file_name, 'w')
    n_channels = 1
    sampwidth = 2
    n_frames = len(data)


    wav_file.setparams((n_channels, sampwidth, sample_rate, n_frames, 'NONE', 'not compressed'))

    for sample in data:
        sample = int(sample)  
        
        sample = max(-32768, min(32767, sample))  

        wav_file.writeframes(struct.pack('h', sample))

    wav_file.close()


def main():
    total_samples = 2000000

    for i in range(21, 26):
        output_file = f"oddech\\generated\\{str(i)}"
        print(output_file)
        breath_data = generate_breath_data(total_samples, output_file)
        save_to_wav(output_file, breath_data)
        print(f"Zapisano wygenerowane dane oddechowe do {output_file}")



if __name__ == "__main__":
    main()
