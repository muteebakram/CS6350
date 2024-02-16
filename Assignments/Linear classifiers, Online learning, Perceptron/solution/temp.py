import numpy as np

bias = 0.001
weights = [
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
]
example = [
    1.0,
    1.0,
    37.0,
    36.0,
    31.0,
    28.0,
    26.0,
    19.0,
    104.319698,
    24.83296,
    9.259328,
    1.932381,
    0.477934,
    0.299354,
    0.185806,
    0.148645,
    0.518568,
    0.102193,
    0.0,
]
actual_label = 1
margin = 1

value = actual_label * (np.dot(weights, example) + bias)
learning_rate = (margin - (actual_label * (np.dot(weights, example) + bias))) / (np.dot(example, example) + 1)

print(value)
print(learning_rate)