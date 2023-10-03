import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

class VoiceCloner:
    def __init__(self, directory):
        self.directory = directory
        self.audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        self.mfccs = []

    def process_audio_files(self):
        for file in self.audio_files:
            y, sr = librosa.load(os.path.join(self.directory, file))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            self.mfccs.append(mfcc)

    def plot_mfccs(self):
        for i, mfcc in enumerate(self.mfccs):
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc, x_axis='time')
            plt.colorbar()
            plt.title(f'MFCC for {self.audio_files[i]}')
            plt.tight_layout()
            plt.show()
'''
# Usage:
cloner = VoiceCloner('audio')
cloner.process_audio_files()
cloner.plot_mfccs()
'''




import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the wav file
sample_rate, data = wavfile.read('audio/70s Electric Piano 11 copy.wav')

# Perform STFT on the audio data
frequencies, times, Zxx = stft(data, fs=sample_rate)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(frequencies, times, np.abs(Zxx))

# Show the plot
plt.show()
