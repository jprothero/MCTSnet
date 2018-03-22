import numpy as np

batch_of_states = np.zeros((128,) + (3, 6, 7))

for state in batch_of_states:
    state[2] += np.random.choice([0, 1])

print(batch_of_states[:3])