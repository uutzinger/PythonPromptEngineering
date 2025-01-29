import numpy as np
import sounddevice as sd

fs = 44100  # Sampling rate (Hz)
duration = 10.0  # seconds
frequency = 2000  # Hz (tone frequency)

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
tone = 0.5 * np.sin(2 * np.pi * frequency * t)  # Generate sine wave

sd.play(tone, samplerate=fs)
sd.wait()