import numpy as np

def generate_music(model, seed, length):
    result = []
    input_seq = seed

    for _ in range(length):
        prediction = model.predict(input_seq)
        result.append(prediction[0][0])
        input_seq = np.roll(input_seq, -1)
        input_seq[-1] = prediction

    return np.array(result)
