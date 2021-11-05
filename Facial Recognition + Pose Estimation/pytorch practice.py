import torch
import numpy as np

# tensors
x = torch.ones(2, 2, dtype=torch.float16)
print(x.size())

a = np.ones(4)
print(a)
b = torch.from_numpy(a)

a += 1
print(a)

b += 1
print(b)
