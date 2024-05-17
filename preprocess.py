import numpy as np
import librosa

def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)  # амплитуда 0.5 для избежания клиппинга
    return t, y

def load_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def prepare_data(y, sequence_length=100):
    X = []
    y = []

    for i in range(len(y) - sequence_length):
        X.append(y[i:i + sequence_length])
        y.append(y[i + sequence_length])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.reshape(y, (y.shape[0], 1))
    return X, y
