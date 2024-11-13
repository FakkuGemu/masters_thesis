import os
import wave


folder_path = "../one_microphone_recordings"


for filename in os.listdir(folder_path):
    
    if filename.startswith("inhale") and filename.endswith(".wav"):
        
        with wave.open(os.path.join(folder_path, filename), 'rb') as wav_file:
            sample_length = wav_file.getnframes()

        txt_filename = filename.replace(".wav", ".txt")
        txt_path = os.path.join(folder_path, txt_filename)
        
        with open(txt_path, 'w') as txt_file:
            txt_file.write(f"wdech 0 {sample_length}")

        print(f"Utworzono plik: {txt_filename} z długością: {sample_length}")
