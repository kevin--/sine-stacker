import soundfile as sf


def write_wave(data, samplerate: number, name: str):
    sf.write(name, data, samplerate)
