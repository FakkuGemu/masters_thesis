from pydub.generators import WhiteNoise
from pydub import AudioSegment



def generate_breath(breath_duration=2000, pause_duration=1000, repetitions=5):
    
    breath_sound = WhiteNoise().to_audio_segment(duration=breath_duration).low_pass_filter(2000).fade_in(500).fade_out(
        500)

    
    pause = AudioSegment.silent(duration=pause_duration)

    
    full_breath = breath_sound + pause + breath_sound

    
    breathing_sequence = full_breath * repetitions

    return breathing_sequence



breath_sound = generate_breath()
breath_sound.export("breath.wav", format="wav")
