#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example for spectral power FFT reduction
"""
import numpy as np
from UliEngineering.SignalProcessing.FFT import simple_serial_spectral_power_fft_reduce
from UliEngineering.SignalProcessing.Simulation import sine_wave
import matplotlib.pyplot as plt

# Generate a signal: 10 Hz sine wave + noise
samplerate = 1000.0
length = 10000 # samples
frequency = 10.0
amplitude = 2.0
noise_level = 0.5

# Generate signal
t = np.arange(length) / samplerate
signal = amplitude * np.sin(2 * np.pi * frequency * t)
noise = np.random.normal(0, noise_level, length)
data = signal + noise

# Compute spectral power
# FFT size 1024
fftsize = 1024
fft = simple_serial_spectral_power_fft_reduce(data, samplerate, fftsize, window="blackman")

print(f"Dominant frequency: {fft.dominant_frequency():.2f} Hz")
print(f"Dominant value (Power): {fft.dominant_value().amplitude:.2f}")
print(f"Expected Power: {amplitude**2:.2f} (attenuated by window)")

# Plot
plt.figure()
plt.plot(fft.frequencies, fft.amplitudes)
plt.title("Spectral Power Density")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [Amplitude^2]")
plt.grid()
# plt.show() # Uncomment to show
