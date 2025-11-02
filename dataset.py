# create a synthetic dataset in line w/ Fleuret (2025)

import random
from typing import List

def generate_synthetic_data(num_samples: int) -> List[str]:
    data = []
    letter_seq_length = 8
    exclamation_prob = 1/16

    for _ in range(num_samples):
        sample = "_" * 64
        letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        prompt = f"{letter}>"
        position = random.randint(0, 64 - letter_seq_length)
        sample = sample[:position] + letter * letter_seq_length + sample[position + letter_seq_length:]
        exlamation_positions = [i for i in range(64) if random.random() < exclamation_prob]
        for pos in exlamation_positions:
            sample = sample[:pos] + "!" + sample[pos + 1:]
        samples = prompt + sample
        data.append(samples)

    return data

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(10)
    for line in synthetic_data[:5]:
        print(line)
