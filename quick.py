import numpy as np

test = list(range(30))

samples = np.random.choice(np.arange(10), 5).astype(int).tolist()
print(test[samples])
