import numpy as np
import sounddevice as sd
from scipy.signal import iirnotch, filtfilt
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg
import sys

# Parameters
fs = 44100  # Sampling rate (Hz)
notch_freq = 2000  # Frequency to filter out (Hz)
quality_factor = 30  # Quality factor for notch filter
chunk_size = 1024  # Number of samples per block

# Design the 60 Hz notch filter
b, a = iirnotch(notch_freq, quality_factor, fs)

# Initialize PyQtGraph application
app = QtWidgets.QApplication([])

# Create the main window
win = pg.GraphicsLayoutWidget(title="Real-Time Audio Filtering")
win.resize(800, 500)

# Create waveform plot
waveform_plot = win.addPlot(title="Time-Domain Signal")
waveform_curve = waveform_plot.plot(pen="y")  # Yellow waveform
waveform_plot.setYRange(-1, 1)

# Create power spectrum plot
win.nextRow()  # Move to the next row in the GUI
spectrum_plot = win.addPlot(title="Power Spectrum")
spectrum_curve = spectrum_plot.plot(pen="c")  # Cyan spectrum
spectrum_plot.setLogMode(x=True, y=True)  # Log frequency scale
spectrum_plot.setLabel("bottom", "Frequency", "Hz")
spectrum_plot.setLabel("left", "Power (dB)")

# Buffers for real-time updates
time_buffer = np.zeros(chunk_size)
signal_buffer = np.zeros(chunk_size)
freqs = np.fft.rfftfreq(chunk_size, d=1/fs)
power_spectrum = np.zeros(len(freqs))

# Callback function for real-time processing
def audio_callback(indata, frames, time, status):
    global signal_buffer, power_spectrum

    if status:
        print(status, flush=True)

    # Apply notch filter
    filtered_signal = filtfilt(b, a, indata[:, 0])

    # Compute power spectrum
    fft_spectrum = np.fft.rfft(filtered_signal)
    power_spectrum = np.abs(fft_spectrum) ** 2  # Squared magnitude

    # Update buffers for live plotting
    signal_buffer = filtered_signal

# Update GUI every frame
def update_plot():
    waveform_curve.setData(signal_buffer)
    spectrum_curve.setData(freqs, power_spectrum)

# Start real-time audio streaming
stream = sd.InputStream(callback=audio_callback, samplerate=fs, channels=1, blocksize=chunk_size)
with stream:
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(30)  # Update every 30ms
    win.show()
    sys.exit(app.exec_())
