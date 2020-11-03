import sys
from torch.utils.data import DataLoader, TensorDataset
import os
import torch.nn as nn
import torch

a = torch.randint(low = 0, high = 5, size = (50, 2))
b = torch.randint(low = 0, high = 2, size = (50,))

print("original a:\t", a)
print("original b:\t", b)

print("shape change:\t", b)

tensor = TensorDataset(a, b)
tensor_loader = DataLoader(tensor, shuffle = True, batch_size = 10)
tensor_iter = iter(tensor_loader)
first = next(tensor_iter)
print(first)


