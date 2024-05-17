from preprocess import generate_sine_wave, prepare_data
from model import create_model, train_model
from generate import generate_music
import soundfile as sf
import os

# Определите базовую директорию
base_dir = os.path.dirname(os.path.abspath(__file__))

# Генерация синусоидальных данных
sr = 22050  # Частота дискретизации
duration = 5.0  # Длительность в секундах
freq = 440.0  # Частота синусоиды (A4)
x, y = generate_sine_wave(freq, sr, duration)

# Подготовка данных для обучения
X, y = prepare_data(y)

# Создание и обучение модели
input_shape = (X.shape[1], 1)
model = create_model(input_shape)
train_model(model, X, y)

# Генерация музыки
seed = X[:100]
generated_music = generate_music(model, seed, 5000)
output_path = os.path.join(base_dir, 'generated_music.wav')
sf.write(output_path, generated_music, sr)

print(f"Generated music saved to {output_path}")
