import torch, numpy as np

k = torch.from_numpy(np.random.randint(low=0, high=10, size=5))
a = torch.from_numpy(np.arange(11))

print(a[k])