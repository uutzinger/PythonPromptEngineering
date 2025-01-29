import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import iirnotch, filtfilt


# Parameters for the sine wave
frequency_signal = 1000  # Hz (Main sine wave frequency)
frequency_noise = 60  # Hz (Noise frequency)
duration = 2.0    # seconds
sample_rate = 44100  # Hz (CD-quality audio)

amplitude_signal = 0.5  # Main signal amplitude
amplitude_noise = amplitude_signal / 3  # Noise amplitude

# Generate time axis
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate sine wave (1000 Hz) and noise (60 Hz)
sine_wave = amplitude_signal * np.sin(2 * np.pi * frequency_signal * t)
noise_wave = amplitude_noise * np.sin(2 * np.pi * frequency_noise * t)


# Plot the sine wave
plt.figure(figsize=(10, 4))
plt.plot(t[:1000], sine_wave[:1000])  # Show the first 1000 samples (~22 ms)
plt.title("1000 Hz Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Play the sine wave
#sd.play(sine_wave, samplerate=sample_rate)
#sd.wait()  # Wait for the sound to finish playing

# Save the sine wave to a file
wave_filename = "sine_wave.wav"
sf.write(wave_filename, sine_wave, sample_rate)
print(f"Sine wave saved as: {wave_filename}")

# Compute Power Spectrum (FFT)
fft_spectrum = np.fft.fft(sine_wave)
frequencies = np.fft.fftfreq(len(fft_spectrum), d=1/sample_rate)
power_spectrum = np.abs(fft_spectrum) ** 2  # Squared magnitude of FFT
log_power_spectrum = 10 * np.log10(power_spectrum + 1e-10) 

# Plot the Power Spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies)//2], log_power_spectrum[:len(frequencies)//2])  # Only show positive frequencies
plt.title("Power Spectrum of 1000 Hz Sine Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.grid()
plt.show()

# Save the power spectrum plot
power_spectrum_filename = "power_spectrum.png"
plt.savefig(power_spectrum_filename)
print(f"Power spectrum saved as: {power_spectrum_filename}")

# Add the noise
# Combine signal and noise
combined_wave = sine_wave + noise_wave

# Save the combined wave as a WAV file
wave_filename = "combined_wave.wav"
sf.write(wave_filename, combined_wave, sample_rate)

# Play the combined signal
# sd.play(combined_wave, samplerate=sample_rate)
# sd.wait()

plt.figure(figsize=(10, 4))
plt.plot(t[:1000], combined_wave[:1000], label="Combined Signal (1000 Hz + 60 Hz Noise)")
plt.plot(t[:1000], sine_wave[:1000], linestyle="dashed", alpha=0.7, label="1000 Hz Signal")
plt.plot(t[:1000], noise_wave[:1000], linestyle="dotted", alpha=0.7, label="60 Hz Noise")
plt.title("Time-Domain Signal (Sine Wave + Noise)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
time_domain_plot_filename = "time_domain_plot.png"
plt.savefig(time_domain_plot_filename)
plt.show()

# Compute Power Spectrum (FFT)
fft_spectrum = np.fft.fft(combined_wave)
frequencies = np.fft.fftfreq(len(fft_spectrum), d=1/sample_rate)
power_spectrum = np.abs(fft_spectrum) ** 2  # Squared magnitude

# Convert power spectrum to dB scale
log_power_spectrum = 10 * np.log10(power_spectrum + 1e-10)  # Avoid log(0) issue

# Plot the Power Spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies)//2], log_power_spectrum[:len(frequencies)//2])  # Only show positive frequencies
plt.title("Power Spectrum of 1000 Hz Sine Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.grid()
plt.show()

# Save the power spectrum plot
power_spectrum_filename = "power_spectrum.png"
plt.savefig(power_spectrum_filename)
print(f"Power spectrum saved as: {power_spectrum_filename}")


# Define notch filter parameters
notch_freq = 60  # Frequency to be removed from signal (Hz)
quality_factor = 30  # Higher value means narrower notch
sample_rate = 44100  # Sampling rate (Hz)

# Design the notch filter
b, a = iirnotch(notch_freq, quality_factor, sample_rate)

# Apply the notch filter to the combined wave
filtered_wave = filtfilt(b, a, combined_wave)

plt.figure(figsize=(10, 4))
plt.plot(t[:1000], filtered_wave[:1000], label="Filtered Signal (1000 Hz + 60 Hz Noise)")
plt.plot(t[:1000], combined_wave[:1000], linestyle="dashed", alpha=0.7, label="1000 + 60Hz Hz Signal")
plt.title("Time-Domain Signal Notchfilter of (Sine Wave + Noise)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
time_domain_plot_filename = "time_domain_plot_filter.png"
plt.savefig(time_domain_plot_filename)
plt.show()

# Compute Power Spectrum (FFT)
fft_spectrum = np.fft.fft(filtered_wave)
frequencies = np.fft.fftfreq(len(fft_spectrum), d=1/sample_rate)
power_spectrum = np.abs(fft_spectrum) ** 2  # Squared magnitude of FFT
log_power_spectrum = 10 * np.log10(power_spectrum + 1e-10)  # Convert to dB scale

# Plot the Power Spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies)//2], log_power_spectrum[:len(frequencies)//2])  # Only show positive frequencies
plt.title("Power Spectrum of Filtered Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.grid()
plt.show()

# Save the power spectrum plot
power_spectrum_filename = "power_spectrum_filter.png"
plt.savefig(power_spectrum_filename)
print(f"Power spectrum saved as: {power_spectrum_filename}")

