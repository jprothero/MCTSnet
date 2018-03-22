import torch
from fractions import gcd

b = torch.randn((3, 6, 7)).unsqueeze(0)
a = torch.randn((128))

print(b.view(-1).numpy().shape)