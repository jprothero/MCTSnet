import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

test = []

a = Variable(torch.rand(5, 3), requires_grad=True)
b = Variable(torch.rand(5, 3), requires_grad=True)
#so c should propagate grad to a and b
c = a * b
test.extend([a])
test.extend([b])
test.extend([c])

del a
del b
del c

pickle.dump(test,
        open("checkpoints/test.p", "wb"))

[a, b, c] = pickle.load(
        open("checkpoints/test.p", "rb"))

d = Variable(torch.randn(5, 3), volatile=True)
loss = F.mse_loss(c, d)


# print(a.grad, b.grad, c.grad, d.grad)
loss.backward()
print(a.grad, b.grad, c.grad, d.grad)